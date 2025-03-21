import os
import torch
from MLP_LSTM.MLN import MLP_LSTM_Network
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch_observe = torch.cat((torch.zeros(2000, 100, 7), torch.zeros(2000, 100, 7)), dim=0)
torch_action = torch.cat((torch.zeros(2000, 100, 3), torch.zeros(2000, 100, 3)), dim=0)
torch_obstacle = torch.cat((torch.zeros(2000, 100, 10, 3), torch.zeros(2000, 100, 10, 3)), dim=0)
torch_mask = torch.cat((torch.zeros(2000, 100, 10), torch.zeros(2000, 100, 10)), dim=0)

data_size = torch_observe.shape[0]
seq_len = torch_obstacle.shape[1]
observe_dim = torch_observe.shape[2]
action_dim = torch_action.shape[2]

max_obstacle = torch_obstacle.shape[2]
obstacle_dim = torch_obstacle.shape[3]

# 数据标准化
observe_mean = torch_observe.mean(dim=0)
observe_std = torch_observe.std(dim=0) + 1e-6

action_mean = torch_action.mean(dim=0)
action_std = torch_action.std(dim=0) + 1e-6

obstacle_mean = torch_obstacle.mean(dim=0)
obstacle_std = torch_obstacle.std(dim=0) + 1e-6

observe_norm = (torch_observe - observe_mean) / (observe_std + 1e-6)
action_norm = (torch_action - action_mean) / (action_std + 1e-6)
obstacle_norm = (torch_obstacle - obstacle_mean) / (obstacle_std + 1e-6)

# Separate dataset in train and val data
n_train = int(0.8 * data_size)
n_eval = int(0.9 * data_size)
train_data = {'observes': observe_norm[:n_train, :, :], 'actions': action_norm[:n_train, :, :], 'obstacles': obstacle_norm[:n_train, :, :, :], 'masks': torch_mask[:n_train, :, :]}
eval_data = {'observes': observe_norm[n_train:n_eval, :, :], 'actions': action_norm[n_train:n_eval, :, :], 'obstacles': obstacle_norm[n_train:n_eval, :, :, :], 'masks': torch_mask[n_train:n_eval, :, :]}

class MLP_LSTM_Dataset(Dataset):
    def __init__(self, split):
        assert split in ['train', 'eval', 'test']
        self.split = split
        self.data = train_data if (split == 'train') or (split == 'eval') else eval_data
        self.data_size = self.data['observes'].shape[0]
        self.seq_len = self.data['observes'].shape[1]

    def __len__(self):
        return self.data['observes'].shape[0]
        # return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(0, self.data_size, (1,))
        observes = self.data['observes'][ix, :, :].view(self.seq_len, observe_dim)
        actions = self.data['actions'][ix, :, :].view(self.seq_len, action_dim)
        obstacles = self.data['obstacles'][ix, :, :, :].view(self.seq_len, max_obstacle, obstacle_dim)
        mask = self.data['masks'][ix, :, :].view(self.seq_len, max_obstacle)
        timesteps = torch.tensor(range(self.seq_len)).view(self.seq_len, 1)

        return observes, actions, obstacles, mask, timesteps, ix

train_dataset = MLP_LSTM_Dataset(split = 'train')
eval_dataset = MLP_LSTM_Dataset(split = 'eval')
observes_i, actions_i, obstacles_i, mask_i, timesteps_i, ix = train_dataset[0]

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
        self.max_obstacle = max_obstacle        # 最大允许观测到的障碍数量
        self.MLP_hidden_size = 128              # MLP 中间层的大小
        self.MLP_output_size = 5                # MLP 输出层的大小
        self.hidden_size = 64                   # LSTM 隐藏层的大小
        self.num_layers = 1                     # LSTM 层数
        self.action_dim = action_dim            # 输出的维度
        self.seq_len = 20                       # 输入序列的长度
        self.action_tanh = False

# 创建config实例
config = Config()

model = MLP_LSTM_Network(config)
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 100
eval_epoch = 50
completed_step = 0

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_loader):
        observes_i, actions_i, obstacles_i, mask_i, timesteps_i, ix = batch
        batch_size = observes_i.shape[0]
        observes_inputs = torch.zeros((batch_size, config.seq_len, observe_dim))
        obstacles_inputs = torch.zeros((batch_size, config.seq_len, max_obstacle, obstacle_dim))
        mask_inputs = torch.zeros((batch_size, config.seq_len, max_obstacle))
        outputs = torch.zeros((batch_size, seq_len, action_dim))
        for timestep in range(seq_len):
            observes_inputs[:, max(-config.seq_len, -(timestep + 1)):, :] = observes_i[:, max(0, timestep - config.seq_len + 1):(timestep + 1), :]
            obstacles_inputs[:, max(-config.seq_len, -(timestep + 1)):, :, :] = obstacles_i[:, max(0, timestep - config.seq_len + 1):(timestep + 1), :, :]
            mask_inputs[:, max(-config.seq_len, -(timestep + 1)):, :] = mask_i[:, max(0, timestep - config.seq_len + 1):(timestep + 1), :]
            output = model(
                observe=observes_inputs,
                obstacle=obstacles_inputs,
                mask=mask_inputs
            )
            output = output[:, -1, :]
            outputs[:, timestep, :] = output
        loss_i = torch.mean((actions_i-outputs) ** 2)
        losses.append(loss_i.item())
    loss = torch.mean(torch.tensor(losses))
    return loss.item()


if __name__ == '__main__':
    for epoch in range(epochs):
        print('第{}轮训练开始，共{}轮'.format(epoch + 1, epochs))
        for step, batch in enumerate(train_loader):
            observes_i, actions_i, obstacles_i, mask_i, timesteps_i, ix = batch
            batch_size = observes_i.shape[0]

            observes_inputs = torch.zeros((batch_size, config.seq_len, observe_dim))
            obstacles_inputs = torch.zeros((batch_size, config.seq_len, max_obstacle, obstacle_dim))
            mask_inputs = torch.zeros((batch_size, config.seq_len, max_obstacle))

            outputs = torch.empty((batch_size, seq_len, config.action_dim))

            for timestep in range(seq_len):
                observes_inputs[:, max(-config.seq_len, -(timestep + 1)):, :] = observes_i[:,max(0, timestep - config.seq_len + 1):(timestep + 1), :]
                obstacles_inputs[:, max(-config.seq_len, -(timestep + 1)):, :, :] = obstacles_i[:, max(0,timestep - config.seq_len + 1):(timestep + 1), :, :]
                mask_inputs[:, max(-config.seq_len, -(timestep + 1)):, :] = mask_i[:,max(0, timestep - config.seq_len + 1):(timestep + 1), :]
                output = model(
                    observe=observes_inputs,
                    obstacle=obstacles_inputs,
                    mask=mask_inputs
                )
                output = output[:, -1, :]
                outputs[:, timestep, :] = output
            loss = torch.mean((outputs - actions_i) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed_step += 1
            if step % 10 == 0:
                print(
                    "Epoch {}/{} ".format(epoch+1, epochs),
                    "| Batch {}/{} ".format(step, len(train_loader)),
                    "| Loss %.4f"%(loss.item())
                )
        if epoch % (eval_epoch) == 0:
            eval_loss = evaluate()
            print('eval loss:', eval_loss)

        if epoch % eval_epoch*10 == 0:
           torch.save(model.state_dict(), f=root_folder+'\MLP_LSTM\saved_files\checkpoints\mist.pth')
           # accelerator.save_state(root_folder + '/transformer/saved_files/checkpoints/checkpoint_rtn_art')
        print('第{}轮训练结束，共{}轮'.format(epoch + 1, epochs))

