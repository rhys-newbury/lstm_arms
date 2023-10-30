import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from pathlib import Path
import typer
from typing import Annotated

app = typer.Typer()

USE_WANDB = True
if USE_WANDB:
    import wandb
class CustomDataset(Dataset):
    def __init__(self, train=True, workspace : Path = Path("/home/worker/smac_transformer/data/")):
        if train:
            l = []
            for i in workspace.glob("*.npy"):
                idx = int(i.name.split("_")[1][0])
                if idx != 6:
                    data_ = np.load(str(i))
                    zero_batch_indices = np.where(np.all(data_ == 0, axis=(1, 2)))

                    # Filter out batches where all elements are zero
                    filtered_data = np.delete(data_, zero_batch_indices, axis=0)
                    l.append(filtered_data)
            data = np.concatenate(l, axis=0)
        else:
            l = []
            for i in workspace.glob("*.npy"):
                idx = int(i.name.split("_")[1][0])
                if idx == 6:
                    data_ = np.load(str(i))
                    zero_batch_indices = np.where(np.all(data_ == 0, axis=(1, 2)))

                    # Filter out batches where all elements are zero
                    filtered_data = np.delete(data_, zero_batch_indices, axis=0)
                    l.append(filtered_data)
            data = np.concatenate(l, axis=0)

        data = torch.tensor(data, dtype=torch.float32, device="cuda")
        self.goals = data[:, 1:, -10:-7]
        self.data = data[:, :-1, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.goals[idx]
        return sample, target


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=6, batch_first=True)

        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, lstm_states=None):
        lstm_out, hidden = self.lstm(x, lstm_states)
        output = self.model(lstm_out)

        return output, hidden

def plot(epoch, test_loader, model, test=True):


    data, target = next(iter(test_loader))
    data = data[:1, :60, :]
    target = target[0, :60, :]
    predicted_total = torch.zeros_like(target, device="cuda")
    hidden = None
    for t in range(60):
        input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
        predicted, hidden = model(input_data, hidden)

        if t > 10:
            data[:, t+1:t+2, -10:-7] = predicted.clone().detach()

        predicted_total[t:t+1, :] = predicted

    predicted_total = predicted_total.squeeze()
    target = target.squeeze()

    # Extracting individual dimensions
    dimension_0 = target[:, 0].detach().cpu().numpy()
    dimension_1 = target[:, 1].detach().cpu().numpy()
    dimension_2 = target[:, 2].detach().cpu().numpy()

    predicted_0 = predicted_total[:, 0].detach().cpu().numpy()
    predicted_1 = predicted_total[:, 1].detach().cpu().numpy()
    predicted_2 = predicted_total[:, 2].detach().cpu().numpy()

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plotting the first dimension
    axs[0].plot(dimension_0, label='Target')
    axs[0].plot(predicted_0, label='Predicted')
    axs[0].set_title('Dimension 0')
    axs[0].legend()

    # Plotting the second dimension
    axs[1].plot(dimension_1, label='Target')
    axs[1].plot(predicted_1, label='Predicted')
    axs[1].set_title('Dimension 1')
    axs[1].legend()

    # Plotting the third dimension
    axs[2].plot(dimension_2, label='Target')
    axs[2].plot(predicted_2, label='Predicted')
    axs[2].set_title('Dimension 2')
    axs[2].legend()

    

    plt.tight_layout()
    # plt.savefig(f"/home/worker/smac_transformer/plot_{epoch}.png")
    if USE_WANDB:
        wandb.log({f"plot_{'test' if test else 'train'}": wandb.Image(plt)})
    plt.close()

@app.command()
def train(workspace : Annotated[Path, typer.Option()] = Path("/home/worker/smac_transformer/data/"),
          hidden_size : Annotated[int, typer.Option()] = 512,
          lr : Annotated[float, typer.Option()] = 0.008,
          batch_size: Annotated[int, typer.Option()]= 512,
          horizon: Annotated[int, typer.Option()]= 30,
          history: Annotated[int, typer.Option()]= 30,
          num_epochs: Annotated[int, typer.Option()] = 100):

    # Define hyperparameters
    input_size = 25  # Number of features in your input
    output_size = 3  # Number of output classes or values

    model = LSTMModel(input_size, hidden_size, output_size)
    model.cuda()
    # Choose a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    shuffle = True   # Set to True if you want to shuffle the data during training    
    train_loader = DataLoader(CustomDataset(train=True, workspace=workspace), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(CustomDataset(train=False, workspace=workspace), batch_size=batch_size, shuffle=False)

    weights = [1] * history + [5] * 10 + [1] * (horizon - 10)

    for epoch in range(num_epochs):
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            hidden = None
            loss = 0
            for t in range(horizon + history):
                input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
                predicted, hidden = model(input_data, hidden)
                loss += criterion(predicted, target[:, t:t+1, :]) * weights[t]
                if t > history:
                    data[:, t+1:t+2, -10:-7] = predicted.clone().detach()
            # print(idx, len(train_loader))
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = loss
        print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}")

        # Test loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                hidden = None
                loss = 0
                for t in range(horizon + history):
                    input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
                    predicted, hidden = model(input_data, hidden)
                    loss += criterion(predicted, target[:, t:t+1, :]) * weights[t]
                    if t > history:
                        data[:, t+1:t+2, -10:-7] = predicted.clone().detach()

                test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}")
        if USE_WANDB:
            wandb.log({
                "Epoch": epoch + 1,
                "Test Loss": test_loss,
                "Train Loss": train_loss,
            })        
        plot(epoch, test_loader, model, test=True)
        plot(epoch, train_loader, model, test=False)


if __name__=="__main__":
    if USE_WANDB:
        wandb.init(project='lstm_arms')
    app()