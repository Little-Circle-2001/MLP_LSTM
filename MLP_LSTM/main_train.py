import os
import torch
import argparse
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from contourpy.util import data

from MLP_LSTM.MLN import MLP_LSTM_Network
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}\n")

# Simulation configuration

state_representation = 'rtn'

# load the data
print("Loading data...")

if state_representation == 'roe':
    torch_observe = torch.cat((torch.load(args.data_dir + '/torch_states_roe_scp.pth', weights_only=True), torch.load(args.data_dir + '/torch_states_roe_cvx.pth', weights_only=True)), dim=0)
    torch_delta_observe_target = torch.cat((torch.load(args.data_dir + '/torch_delta_state_roe_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_delta_state_roe_scp.pth', weights_only=True)), dim=0)
else:
    torch_observe = torch.cat((torch.load(args.data_dir + '/torch_states_rtn_scp.pth', weights_only=True), torch.load(args.data_dir + '/torch_states_rtn_cvx.pth', weights_only=True)), dim=0)
    torch_delta_observe_target = torch.cat((torch.load(args.data_dir + '/torch_delta_state_rtn_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_delta_state_rtn_scp.pth', weights_only=True)), dim=0)

torch_obstacle = torch.cat((torch.load(args.data_dir + '/torch_delta_state_rtn_cvx_obs.pth', weights_only=True), torch.load(args.data_dir + '/torch_delta_state_rtn_scp_obs.pth', weights_only=True)), dim=0)
torch_koz_vector = torch.cat((torch.load(args.data_dir + '/torch_koz_vector_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_koz_vector_scp.pth', weights_only=True)), dim=0)
torch_action = torch.cat((torch.load(args.data_dir + '/torch_actions_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_actions_scp.pth', weights_only=True)), dim=0)
torch_mask = torch.cat((torch.load(args.data_dir + '/torch_mask.pth', weights_only=True), torch.load(args.data_dir + '/torch_mask.pth', weights_only=True)), dim=0)

data_size = torch_observe.shape[0]
seq_len = torch_observe.shape[1]
observe_dim = torch_observe.shape[-1]
obstacle_dim = torch_obstacle.shape[-1] + 1
max_num_obs = torch_obstacle.shape[2]
action_dim = torch_action.shape[2]

print('Completed\n')

# 数据标准化
observe_mean = torch_observe.mean(dim=0)
observe_std = torch_observe.std(dim=0) + 1e-6

delta_observe_target_mean = torch_delta_observe_target.mean(dim=0)
delta_observe_target_std = torch_delta_observe_target.std(dim=0) + 1e-6

obstacle_mean = torch_obstacle.mean(dim=0)
obstacle_std = torch_obstacle.std(dim=0) + 1e-6

action_mean = torch_action.mean(dim=0)
action_std = torch_action.std(dim=0) + 1e-6

observe_norm = (torch_observe - observe_mean) / (observe_std + 1e-6)
delta_observe_target_norm = (torch_delta_observe_target - delta_observe_target_mean) / (delta_observe_target_std + 1e-6)
delta_obstacle_norm = (torch_obstacle - obstacle_mean) / (obstacle_std + 1e-6)
obstacle_norm = torch.cat((delta_obstacle_norm, torch_koz_vector), dim=-1)
action_norm = (torch_action - action_mean) / (action_std + 1e-6)

data_stats = {
    'states_mean': observe_mean,
    'states_std': observe_std,
    'delta_states_target_mean': delta_observe_target_mean,
    'delta_states_target_std': delta_observe_target_std,
    'obstacle_mean': obstacle_mean,
    'obstacle_std': obstacle_std,
    'actions_mean': action_mean,
    'actions_std': action_std,
}

np.savez_compressed(
    '..\dataset\dataset-rpod-stats.npz',
    states_std=data_stats['states_std'],
    states_mean=data_stats['states_mean'],
    delta_states_target_mean=data_stats['delta_states_target_mean'],
    delta_states_target_std=data_stats['delta_states_target_std'],
    obstacle_mean=data_stats['obstacle_mean'],
    obstacle_std=data_stats['obstacle_std'],
    actions_mean=data_stats['actions_mean'],
    actions_std=data_stats['actions_std']
)

# Separate dataset in train and val data
n_train = int(0.8 * data_size)
n_eval = int(0.9 * data_size)

train_data = {
    'observes': observe_norm[:n_train, :, :],
    'observes_target': delta_observe_target_norm[:n_train, :, :],
    'obstacles': obstacle_norm[:n_train, :, :, :],
    'masks': torch_mask[:n_train, :, :],
    'actions': action_norm[:n_train, :, :]
}
eval_data = {
    'observes': observe_norm[n_train:n_eval, :, :],
    'observes_target': delta_observe_target_norm[n_train:n_eval, :, :],
    'obstacles': obstacle_norm[n_train:n_eval, :, :, :],
    'masks': torch_mask[n_train:n_eval, :, :],
    'actions': action_norm[n_train:n_eval, :, :]
}
test_data = {
    'observes': observe_norm[n_eval:, :, :],
    'observes_target': delta_observe_target_norm[n_eval:, :, :],
    'obstacles': obstacle_norm[n_eval:, :, :, :],
    'masks': torch_mask[n_eval:, :, :],
    'actions': action_norm[n_eval:, :, :]
}

class MLP_LSTM_Dataset(Dataset):
    def __init__(self, split):
        assert split in ['train', 'eval', 'test']
        self.split = split
        self.data = train_data if (split == 'train') else eval_data
        self.data_size = self.data['observes'].shape[0]
        self.seq_len = self.data['observes'].shape[1]

    def __len__(self):
        return self.data['observes'].shape[0]
        # return len(self.data)

    def __getitem__(self, idx):
        # ix = torch.randint(0, self.data_size, (1,))
        ix = idx
        observes = self.data['observes'][ix, :, :].view(self.seq_len, observe_dim)
        delta_observes_target = self.data['observes_target'][ix, :, :].view(self.seq_len, observe_dim)
        obstacles = self.data['obstacles'][ix, :, :, :].view(self.seq_len, max_num_obs, obstacle_dim)
        actions = self.data['actions'][ix, :, :].view(self.seq_len, action_dim)
        mask = self.data['masks'][ix, :, :].view(self.seq_len, max_num_obs)
        time = torch.tensor(range(self.seq_len)).view(self.seq_len, 1)
        timesteps = (self.seq_len - 1 - time) / (self.seq_len - 1)

        return observes, delta_observes_target, obstacles, actions, mask, timesteps, ix

train_dataset = MLP_LSTM_Dataset(split='train')
eval_dataset = MLP_LSTM_Dataset(split='eval')
observes_i, delta_observes_target_i, obstacles_i, actions_i, mask_i, timesteps_i, ix = train_dataset[0]

train_loader = DataLoader(
    train_dataset,
    # sampler=torch.utils.data.RandomSampler(
    #     train_dataset, replacement=True, num_samples=int(6400)),
    shuffle=True,
    pin_memory=True,
    batch_size=64,
    num_workers=0,
    drop_last=False
)
eval_loader = DataLoader(
    eval_dataset,
    shuffle=True,
    pin_memory=True,
    batch_size=64,
    num_workers=0,
    drop_last=False
    )

class Config:
    def __init__(self):
        self.observe_dim = observe_dim          # 输入特征的维度
        self.obstacle_dim = obstacle_dim        # 障碍的维度
        self.max_obstacle = max_num_obs         # 最大允许观测到的障碍数量
        self.MLP_hidden_size = 128              # MLP 中间层的大小
        self.MLP_output_size = 5                # MLP 输出层的大小
        self.hidden_size = 128                  # LSTM 隐藏层的大小
        self.num_layers = 2                     # LSTM 层数
        self.action_dim = action_dim + observe_dim            # 输出的维度
        self.control_dim = action_dim
        self.seq_len = 30                       # 输入序列的长度
        self.action_tanh = False

# 创建config实例
config = Config()

model = MLP_LSTM_Network(config)
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 1000
eval_epoch = 50
completed_step = 0
loss_list = []

def evaluate():
    model.eval()
    losses = []
    losses_control = []
    losses_state = []
    for step, batch in enumerate(eval_loader):
        # observes_i, actions_i, obstacles_i, mask_i, timesteps_i, ix = batch
        observes_i, delta_observes_target_i, obstacles_i, actions_i, mask_i, timesteps_i, ix = batch
        batch_size = observes_i.shape[0]

        observes_inputs = torch.empty((batch_size, seq_len, observe_dim))
        delta_observes_target_inputs = torch.empty((batch_size, seq_len, observe_dim))
        obstacles_inputs = torch.empty((batch_size, seq_len, config.max_obstacle, obstacle_dim))
        mask_inputs = torch.empty((batch_size, seq_len, max_num_obs))
        actions_inputs = torch.empty((batch_size, seq_len, action_dim))
        timestep_inputs = torch.zeros((batch_size, seq_len, 1))

        observes_inputs[:, :, :] = observes_i
        delta_observes_target_inputs[:, :, :] = delta_observes_target_i
        obstacles_inputs[:, :, :, :] = obstacles_i
        mask_inputs[:, :, :] = mask_i
        actions_inputs[:, :, :] = torch.cat((torch.zeros(batch_size, 1, action_dim), actions_i[:, :-1, :]), dim=1)
        timestep_inputs[:, :, :] = timesteps_i

        outputs = torch.empty((batch_size, seq_len, config.action_dim))

        output = model(
            observe=observes_inputs,
            delta_observe_target=delta_observes_target_inputs,
            obstacle=obstacles_inputs,
            mask=mask_inputs,
            action=actions_inputs,
            time=timestep_inputs
        )
        outputs[:, :, :] = output

        control = outputs[:, :, :3]
        state = outputs[:, :, 3:]

        loss_i_control = torch.mean((control - actions_i) ** 2)
        loss_i_state = torch.mean((state[:, :-1, :] - observes_i[:, 1:, :]) ** 2)
        loss_i = loss_i_control + loss_i_state
        losses_control.append(loss_i_control.item())
        losses_state.append(loss_i_state.item())
        losses.append(loss_i.item())

    loss_control = torch.mean(torch.tensor(losses_control))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss = torch.mean(torch.tensor(losses))
    return loss_control.item(), loss_state.item(), loss.item()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print('第{}轮训练开始，共{}轮'.format(epoch + 1, epochs))
        for step, batch in enumerate(train_loader):
            observes_i, delta_observes_target_i, obstacles_i, actions_i, mask_i, _, ix = batch
            batch_size = observes_i.shape[0]

            observes_inputs = torch.empty((batch_size, seq_len, observe_dim))
            delta_observes_target_inputs = torch.empty((batch_size, seq_len, observe_dim))
            obstacles_inputs = torch.empty((batch_size, seq_len, config.max_obstacle, obstacle_dim))
            mask_inputs = torch.empty((batch_size, seq_len, max_num_obs))
            actions_inputs = torch.empty((batch_size, seq_len, action_dim))
            timestep_inputs = torch.zeros((batch_size, seq_len, 1))

            observes_inputs[:, :, :] = observes_i
            delta_observes_target_inputs[:, :, :] = delta_observes_target_i
            obstacles_inputs[:, :, :, :] = obstacles_i
            mask_inputs[:, :, :] = mask_i
            actions_inputs[:, :, :] = torch.cat((torch.zeros(batch_size, 1, action_dim), actions_i[:, :-1, :]), dim=1)
            timestep_inputs[:, :, :] = timesteps_i

            outputs = torch.empty((batch_size, seq_len, config.action_dim))

            output = model(
                observe=observes_inputs,
                delta_observe_target=delta_observes_target_inputs,
                obstacle=obstacles_inputs,
                mask=mask_inputs,
                action=actions_inputs,
                time=timestep_inputs
            )

            outputs[:, :, :] = output

            control = outputs[:, :, :3]
            state = outputs[:, :, 3:]

            loss_control = torch.mean((control[:, :, :] - actions_i[:, :, :]) ** 2)
            loss_state = torch.mean((state[:, :-1, :] - observes_inputs[:, 1:, :]) ** 2)

            loss = loss_control + loss_state
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} ",
                    f"| Batch {step}/{len(train_loader)} ",
                    f"| Loss {loss.item()}"
                )
        if (epoch + 1) % (eval_epoch) == 0:
            loss_control, loss_state, eval_loss = evaluate()
            print(
                f"control loss: {loss_control} ",
                f"| state loss: {loss_state} ",
                f"| eval loss: {eval_loss}"
            )

        if (epoch + 1) % eval_epoch*10 == 0:
           torch.save(model.state_dict(), f=root_folder+'\MLP_LSTM\saved_files\checkpoints\checkpoint_MLP_LSTM_v8.pth')
        print('第{}轮训练结束，共{}轮'.format(epoch + 1, epochs))

plt.plot(loss_list, label='loss', color='blue', linestyle='-', linewidth=1)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()