import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# from torch.utils.data import random_split
from pathlib import Path
import typer
from typing import Annotated
import torch.nn.functional as F
import asyncio
from typing import List, Tuple, Optional

app = typer.Typer()

device = "cuda"
USE_WANDB = False
if USE_WANDB:
    import wandb

JOINT_VEL_RANGE = (7, 14)
JOINT_POS_RANGE = (0, 7)
POS_RANGE = (14, 17)


vel_nomralize = None
pos_normalize = None
joint_pos_normalize = None
action_normalize = None


class CustomDataset(Dataset):
    def __init__(
        self,
        train=True,
        workspace: Path = Path("/home/worker/smac_transformer/new_data/"),
        normalize=True,
    ):
        global vel_nomralize
        global pos_normalize
        global joint_pos_normalize
        global action_normalize
        if train:
            data_list = []
            for i in workspace.glob("*.npy"):
                idx = int(i.name.split("_")[1])
                idx = idx if idx < 1000 else idx // 1000
                if idx == 1:
                    data_ = np.load(str(i))
                    zero_batch_indices = np.where(np.all(data_ == 0, axis=(1, 2)))

                    # Filter out batches where all elements are zero
                    filtered_data = np.delete(data_, zero_batch_indices, axis=0)
                    data_list.append(filtered_data)
            data = np.concatenate(data_list, axis=0)
            data = data[: int(data.shape[0] * 0.8), :, :]
        else:
            data_list = []
            for i in workspace.glob("*.npy"):
                idx = int(i.name.split("_")[1])
                idx = idx if idx < 1000 else idx // 1000
                if idx == 1:
                    data_ = np.load(str(i))
                    zero_batch_indices = np.where(np.all(data_ == 0, axis=(1, 2)))

                    # Filter out batches where all elements are zero
                    filtered_data = np.delete(data_, zero_batch_indices, axis=0)
                    data_list.append(filtered_data)
            data = np.concatenate(data_list, axis=0)
            data = data[int(data.shape[0] * 0.8) :, :, :]

        data = torch.tensor(data, dtype=torch.float32, device=device)  # [:, :, 1:]

        if vel_nomralize is None and normalize:
            vel_nomralize = torch.amax(
                torch.abs(data[:, :, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]]),
                axis=(0, 1),
            )
            pos_normalize = torch.amax(
                torch.abs(data[:, :, POS_RANGE[0] : POS_RANGE[1]]), axis=(0, 1)
            )
            joint_pos_normalize = torch.amax(
                torch.abs(data[:, :, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]]),
                axis=(0, 1),
            )
            action_normalize = torch.amax(
                torch.abs(data[:, :, POS_RANGE[1] :]), axis=(0, 1)
            )

        if normalize:
            data[:, :, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]] /= vel_nomralize
            data[:, :, POS_RANGE[0] : POS_RANGE[1]] /= pos_normalize
            data[:, :, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]] /= joint_pos_normalize
            data[:, :, POS_RANGE[1] :] /= action_normalize

        self.goals = data[:, 1:, POS_RANGE[0] : POS_RANGE[1]]
        self.position_goal = data[:, 1:, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]]
        self.velocity_goal = data[:, 1:, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]]

        self.data = data[:, :-1, :]

        self.input_size = self.data.shape[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].to(device)
        target_goal = self.goals[idx].to(device)
        target_vel = self.velocity_goal[idx].to(device)
        target_pos = self.position_goal[idx].to(device)
        return sample, (target_goal, target_pos, target_vel)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=1):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

        self.after_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LeakyReLU(),
        )

        self.position_model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.velocity = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 7),
        )

        self.joint_pos = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 7),
        )

    def forward(
        self,
        x,
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        calc_all=False,
        calc_pos=False,
        calc_joint_pos=False,
        calc_vel=False,
    ):
        lstm_out, hidden = self.lstm(x, lstm_states)

        attention_output, _ = self.attention(lstm_out, lstm_out, lstm_out)

        combined_output = F.leaky_relu(lstm_out + attention_output, negative_slope=0.01)

        combined_output = F.leaky_relu(self.after_attention(combined_output))

        position_output, velocity_output, joint_pos_output = None, None, None

        if calc_all or calc_pos:
            position_output = self.position_model(combined_output)
        if calc_all or calc_joint_pos:
            joint_pos_output = self.joint_pos(combined_output)
        if calc_all or calc_vel:
            velocity_output = self.velocity(combined_output)

        return (position_output, joint_pos_output, velocity_output), hidden


def plot(
    epoch, test_loader, model, history, horizon, number=3, test=True, use_gt=False
):
    for idx, (data, (target_pos, target_joint_pos, target_vel)) in enumerate(
        test_loader
    ):
        if idx == number:
            break

        data = data[:1, : history + horizon, :]
        target_pos = target_pos[0, : history + horizon, :]
        predicted_total = torch.zeros_like(target_pos, device=device)
        hidden = None

        for t in range(horizon + history):
            input_data = data[
                :, t : t + 1, :
            ].clone()  # Clone the tensor before feeding it to the model
            (pos, joint_pos, vel), hidden = model(input_data, hidden, calc_all=True)

            if t > history:
                data[
                    :, t + 1 : t + 2, POS_RANGE[0] : POS_RANGE[1]
                ] = pos.clone().detach()
                data[
                    :, t + 1 : t + 2, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]
                ] = joint_pos.clone().detach()
                data[
                    :, t + 1 : t + 2, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]
                ] = vel.clone().detach()

            predicted_total[t : t + 1, :] = pos

        predicted_total = predicted_total.squeeze()
        target_pos = target_pos.squeeze()

        # Extracting individual dimensions
        dimension_0 = target_pos[:, 0].detach().cpu().numpy()
        dimension_1 = target_pos[:, 1].detach().cpu().numpy()
        dimension_2 = target_pos[:, 2].detach().cpu().numpy()

        predicted_0 = predicted_total[:, 0].detach().cpu().numpy()
        predicted_1 = predicted_total[:, 1].detach().cpu().numpy()
        predicted_2 = predicted_total[:, 2].detach().cpu().numpy()

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        # Plotting the first dimension
        axs[0].plot(dimension_0, label="Target Position")
        axs[0].plot(predicted_0, label="Predicted Position")
        axs[0].set_title("X")
        axs[0].legend()

        # Plotting the second dimension
        axs[1].plot(dimension_1, label="Target Position")
        axs[1].plot(predicted_1, label="Predicted Position")
        axs[1].set_title("Y")
        axs[1].legend()

        # Plotting the third dimension
        axs[2].plot(dimension_2, label="Target Position")
        axs[2].plot(predicted_2, label="Predicted Position")
        axs[2].set_title("Z")
        axs[2].legend()

        axs[0].axvline(x=history, color="r", linestyle="-")
        axs[1].axvline(x=history, color="r", linestyle="-")
        axs[2].axvline(x=history, color="r", linestyle="-")

        plt.tight_layout()
        # plt.savefig(f"/home/worker/smac_transformer/plot_{epoch}.png")
        if USE_WANDB:
            plot_name = (
                f"plot_{idx}_{'test' if test else 'train'}_"
                f"{'gt' if use_gt else 'no_gt'}"
            )
            wandb.log(
                {
                    "Epoch": epoch,
                    plot_name: wandb.Image(plt),
                }
            )
        plt.close()


def calc_loss(
    history: int,
    horizon: int,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    weights: List[int],
    hidden: Tuple[torch.Tensor, torch.Tensor],
    enable_grad=True,
):
    async def _inner(
        data,
        target,
        loss,
        calc_pos=False,
        calc_vel=False,
        calc_joint_pos=False,
        data_range=[0, 0],
        result_idx=0,
    ):
        with torch.set_grad_enabled(enable_grad):
            t = history
            input_data = data[
                :, t : t + 1, :
            ].clone()  # Clone the tensor before feeding it to the model
            result, hidden_pos = model(
                input_data,
                hidden,
                calc_pos=calc_pos,
                calc_vel=calc_vel,
                calc_joint_pos=calc_joint_pos,
            )
            value = result[result_idx]
            data[
                :, t + 1 : t + 2, data_range[0] : data_range[1]
            ] = value.clone().detach()
            loss += criterion(value, target[:, t : t + 1, :]) * weights[t]

            for t in range(history + 1, history + horizon):
                # We need three copies of the hidden state
                # for the three copies of the data
                input_data = data[
                    :, t : t + 1, :
                ].clone()  # Clone the tensor before feeding it to the model
                result, hidden_pos = model(
                    input_data,
                    hidden_pos,
                    calc_pos=calc_pos,
                    calc_vel=calc_vel,
                    calc_joint_pos=calc_joint_pos,
                )
                value = result[result_idx]

                data[
                    :, t + 1 : t + 2, data_range[0] : data_range[1]
                ] = value.clone().detach()
                loss += criterion(value, target[:, t : t + 1, :]) * weights[t]
            return loss

    return _inner


def calc_all(
    history: int,
    horizon: int,
    data: torch.Tensor,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    target_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
    target_vel: torch.Tensor,
    weights: List[int],
    pos_loss: torch.Tensor,
    joint_pos_loss: torch.Tensor,
    vel_loss: torch.Tensor,
    hidden: Tuple[torch.Tensor, torch.Tensor],
    enable_grad: bool = True,
):
    with torch.set_grad_enabled(enable_grad):
        t = history
        input_data = data[
            :, t : t + 1, :
        ].clone()  # Clone the tensor before feeding it to the model
        (goal_pos, joint_pos, joint_vel), hidden = model(
            input_data, hidden, calc_all=True
        )

        data[:, t + 1 : t + 2, POS_RANGE[0] : POS_RANGE[1]] = goal_pos.clone().detach()
        data[
            :, t + 1 : t + 2, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]
        ] = joint_pos.clone().detach()
        data[
            :, t + 1 : t + 2, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]
        ] = joint_vel.clone().detach()

        pos_loss += criterion(goal_pos, target_pos[:, t : t + 1, :]) * weights[t]
        joint_pos_loss += (
            criterion(joint_pos, target_joint_pos[:, t : t + 1, :]) * weights[t]
        )
        vel_loss += criterion(joint_vel, target_vel[:, t : t + 1, :]) * weights[t]

        for t in range(history + 1, history + horizon):
            input_data_joint = data[
                :, t : t + 1, :
            ].clone()  # Clone the tensor before feeding it to the model
            (goal_pos, joint_pos, joint_vel), hidden = model(
                input_data_joint, hidden, calc_all=True
            )
            data[
                :, t + 1 : t + 2, POS_RANGE[0] : POS_RANGE[1]
            ] = goal_pos.clone().detach()
            data[
                :, t + 1 : t + 2, JOINT_POS_RANGE[0] : JOINT_POS_RANGE[1]
            ] = joint_pos.clone().detach()
            data[
                :, t + 1 : t + 2, JOINT_VEL_RANGE[0] : JOINT_VEL_RANGE[1]
            ] = joint_vel.clone().detach()

            pos_loss += criterion(goal_pos, target_pos[:, t : t + 1, :]) * weights[t]
            joint_pos_loss += (
                criterion(joint_pos, target_joint_pos[:, t : t + 1, :]) * weights[t]
            )
            vel_loss += criterion(joint_vel, target_vel[:, t : t + 1, :]) * weights[t]

        return pos_loss, joint_pos_loss, vel_loss


async def train_(
    workspace: Path,
    hidden_size: int,
    lr: float,
    batch_size: int,
    horizon: int,
    history: int,
    num_epochs: int,
    beta: float,
    finetune_epochs: int,
):
    shuffle = True  # Set to True if you want to shuffle the data during training
    train_loader = DataLoader(
        CustomDataset(train=True, workspace=workspace),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        CustomDataset(train=False, workspace=workspace),
        batch_size=batch_size,
        shuffle=False,
    )

    # Define hyperparameters
    input_size = train_loader.dataset.input_size  # Number of features in your input
    output_size = 3  # Number of output classes or values

    model = LSTMModel(input_size, hidden_size, output_size)
    model.to(device=device)
    # Choose a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    weights = [1] * history + [5] * horizon

    for epoch in range(num_epochs + finetune_epochs):
        model.train()
        total_pos_loss = 0
        total_vel_loss = 0
        total_joint_pos_loss = 0
        for idx, (data, (target_goal, target_pos, target_vel)) in enumerate(
            train_loader
        ):
            hidden = None
            pos_loss = 0
            joint_pos_loss = 0
            vel_loss = 0

            data_pos = data.detach().clone()
            data_joint = data.detach().clone()
            data_vel = data.detach().clone()

            for t in range(history):
                print(t)
                input_data = data[:, t : t + 1, :].clone()
                # Use the input data which is the data cloned.
                (pos, joint_pos, vel), hidden = model(input_data, hidden, calc_all=True)
                # We calculate the loss for each of the branches!
                pos_loss += criterion(pos, target_goal[:, t : t + 1, :]) * weights[t]
                joint_pos_loss += (
                    criterion(joint_pos, target_pos[:, t : t + 1, :]) * weights[t]
                )
                vel_loss += criterion(vel, target_vel[:, t : t + 1, :]) * weights[t]

            if epoch < num_epochs:
                general_calc = calc_loss(
                    history,
                    horizon,
                    model,
                    criterion,
                    weights,
                    hidden,
                    enable_grad=True,
                )
                tasks = [
                    general_calc(
                        data_pos,
                        target_goal,
                        pos_loss,
                        calc_pos=True,
                        data_range=POS_RANGE,
                        result_idx=0,
                    ),
                    general_calc(
                        data_joint,
                        target_pos,
                        joint_pos_loss,
                        calc_joint_pos=True,
                        data_range=JOINT_POS_RANGE,
                        result_idx=1,
                    ),
                    general_calc(
                        data_vel,
                        target_vel,
                        vel_loss,
                        calc_vel=True,
                        data_range=JOINT_VEL_RANGE,
                        result_idx=2,
                    ),
                ]
                pos_loss, joint_pos_loss, vel_loss = await asyncio.gather(*tasks)

            else:
                pos_loss, joint_pos_loss, vel_loss = calc_all(
                    history,
                    horizon,
                    data_pos,
                    model,
                    criterion,
                    target_goal,
                    target_pos,
                    target_vel,
                    weights,
                    pos_loss,
                    joint_pos_loss,
                    vel_loss,
                    hidden,
                )

            loss = pos_loss + joint_pos_loss + vel_loss

            total_pos_loss += pos_loss
            total_joint_pos_loss += joint_pos_loss
            total_vel_loss += vel_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_pos_loss /= len(train_loader)
        total_joint_pos_loss /= len(train_loader)
        total_vel_loss /= len(train_loader)

        train_loss = total_pos_loss + total_joint_pos_loss + total_vel_loss

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

        # Test loop
        model.eval()
        test_loss = 0.0
        test_total_pos_loss = 0
        test_total_vel_loss = 0
        test_total_joint_pos_loss = 0
        with torch.no_grad():
            for data, (target_goal, target_pos, target_vel) in test_loader:
                hidden = None
                pos_loss = 0
                joint_pos_loss = 0
                vel_loss = 0

                data_pos = data.detach().clone()
                data_joint = data.detach().clone()
                data_vel = data.detach().clone()

                for t in range(history):
                    input_data = data[:, t : t + 1, :].clone()
                    # Use the input data which is the data cloned.
                    (pos, joint_pos, vel), hidden = model(
                        input_data, hidden, calc_all=True
                    )
                    # We calculate the loss for each of the branches!
                    pos_loss += (
                        criterion(pos, target_goal[:, t : t + 1, :]) * weights[t]
                    )
                    joint_pos_loss += (
                        criterion(joint_pos, target_pos[:, t : t + 1, :]) * weights[t]
                    )
                    vel_loss += criterion(vel, target_vel[:, t : t + 1, :]) * weights[t]

                general_calc = calc_loss(
                    history,
                    horizon,
                    model,
                    criterion,
                    weights,
                    hidden,
                    enable_grad=True,
                )
                tasks = [
                    general_calc(
                        data_pos,
                        target_goal,
                        calc_pos=True,
                        data_range=POS_RANGE,
                        result_idx=0,
                    ),
                    general_calc(
                        data_joint,
                        target_pos,
                        calc_joint_pos=True,
                        data_range=JOINT_POS_RANGE,
                        result_idx=1,
                    ),
                    general_calc(
                        data_vel,
                        target_vel,
                        calc_vel=True,
                        data_range=JOINT_VEL_RANGE,
                        result_idx=2,
                    ),
                ]
                pos_loss, joint_pos_loss, vel_loss = await asyncio.gather(*tasks)

                pos_loss, joint_pos_loss, vel_loss = await asyncio.gather(*tasks)
                loss = pos_loss + joint_pos_loss + vel_loss
                test_loss += loss.item()
                test_total_pos_loss += pos_loss
                test_total_joint_pos_loss += joint_pos_loss
                test_total_vel_loss += vel_loss

        test_total_pos_loss /= len(test_loader)
        test_total_joint_pos_loss /= len(test_loader)
        test_total_vel_loss /= len(test_loader)
        test_loss /= len(test_loader)

        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}")
        if USE_WANDB:
            wandb.log(
                {
                    "Epoch": epoch + 1,
                    "Test Loss": test_loss,
                    "Train Loss": train_loss,
                    "test_total_pos_loss": test_total_pos_loss,
                    "test_total_joint_pos_loss": test_total_joint_pos_loss,
                    "test_total_vel_loss": test_total_vel_loss,
                    "train_total_pos_loss": total_pos_loss,
                    "train_total_joint_pos_loss": total_joint_pos_loss,
                    "train_total_vel_loss": total_vel_loss,
                }
            )
        # fmt: off
        plot(epoch,test_loader,model,history,horizon,number=3,test=True,use_gt=False)
        plot(epoch,train_loader,model,history,horizon,number=3,test=False,use_gt=False)
        plot(epoch,test_loader,model,history,horizon,number=3,test=True,use_gt=True)
        plot(epoch,train_loader,model,history,horizon,number=3,test=False,use_gt=True)
        # fmt: on


@app.command()
def train(
    workspace: Annotated[Path, typer.Option()] = Path(
        "/home/worker/smac_transformer/new_data/"
    ),
    hidden_size: Annotated[int, typer.Option()] = 512,
    lr: Annotated[float, typer.Option()] = 0.008,
    batch_size: Annotated[int, typer.Option()] = 512,
    horizon: Annotated[int, typer.Option()] = 30,
    history: Annotated[int, typer.Option()] = 30,
    num_epochs: Annotated[int, typer.Option()] = 10,
    finetune_epochs: Annotated[int, typer.Option()] = 1000,
    beta: Annotated[float, typer.Option()] = 0.05,
):
    asyncio.run(
        train_(
            workspace,
            hidden_size,
            lr,
            batch_size,
            horizon,
            history,
            num_epochs,
            beta,
            finetune_epochs,
        )
    )


if __name__ == "__main__":
    if USE_WANDB:
        wandb.init(project="lstm_arms")
    app()
