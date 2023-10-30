import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math

class CustomDataset(Dataset):
    def __init__(self, train=True):
        if train:
            data1 = np.load("data.npy")[400:, :, :]
            data2 = np.load("output_1002.npy")[:, :, :]
            data3 = np.load("output_1003.npy")[:, :, :]


            data4 = np.load("output_3001.npy")[:400, :, :]
            data5 = np.load("output_3002.npy")[:400, :, :]
            data6 = np.load("output_3003.npy")[:400, :, :]
            data7 = np.load("output_3004.npy")[:400, :, :]
            data8 = np.load("output_3005.npy")[:400, :, :]

            # data1 = np.load("output_6001.npy")[:400, :, :]    
            # data2 = np.load("output_2001.npy")[:400, :, :]
            # data3 = np.load("output_3001.npy")[:400, :, :]
            # data4 = np.load("output_4001.npy")[:400, :, :]
            # data5 = np.load("output_5001.npy")[:400, :, :]
            data = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8), axis=0)

        else:
            data1 = np.load("data.npy")[:400, :, :]
            data9 = np.load("output_3006.npy")[:400, :, :]
            data = np.concatenate((data1, data9), axis=0)

        data = torch.tensor(data, dtype=torch.float32, device="cuda")
        self.goals = data[:, 1:, -10:-7]
        self.data = data[:, :-1, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.goals[idx]
        return sample, target

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
       

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def plot(epoch, test_loader, model):


    data, target = next(iter(test_loader))
    data = data[:1, :50, :]
    target = target[0, :50, :]
    predicted_total = torch.zeros_like(target, device="cuda")
    hidden = None
    for t in range(50):
        input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
        predicted, hidden = model(input_data, hidden)
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
    plt.savefig(f"/home/worker/smac_transformer/plot_{epoch}.png")
    plt.close()

def train():
    # Define hyperparameters
    input_size = 25  # Number of features in your input
    hidden_size = 128  # Number of LSTM units (hidden states)
    output_size = 3  # Number of output classes or values

    model = TransAm(input_size, hidden_size, output_size)
    model.cuda()
    # Choose a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    batch_size = 512  # You can adjust this based on your needs
    shuffle = True   # Set to True if you want to shuffle the data during training

    horizon = 30
    train_loader = DataLoader(CustomDataset(train=True), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(CustomDataset(train=False), batch_size=batch_size, shuffle=False)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            hidden = None
            loss = 0
            for t in range(horizon):
                input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
                predicted, hidden = model(input_data, hidden)
                loss += criterion(predicted, target[:, t:t+1, :])
                data[:, t+1:t+2, -10:-7] = predicted.clone().detach()
            # print(idx, len(train_loader))
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}")

        # Test loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                hidden = None
                loss = 0
                for t in range(horizon):
                    input_data = data[:, t:t+1, :].clone()  # Clone the tensor before feeding it to the model
                    predicted, hidden = model(input_data, hidden)
                    loss += criterion(predicted, target[:, t:t+1, :])
                    data[:, t+1:t+2, -10:-7] = predicted.clone().detach()

                test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}")
        plot(epoch, test_loader, model)

if __name__=="__main__":
    train()