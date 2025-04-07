import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator
from transformer.art import AutonomousRendezvousTransformer

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *

parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
use_lr_scheduler = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Running on device: {device}\n")

# Simulation configuration

state_representation = 'rtn'

# load the data
print('Loading data...', end='')

    
if state_representation == 'roe':
    torch_states = torch.cat((torch.load(args.data_dir + '/torch_states_roe_scp.pth'),torch.load(args.data_dir + '/torch_states_roe_cvx.pth')),0)
else:
    torch_states = torch.cat((torch.load(args.data_dir + '/torch_states_rtn_scp.pth'),torch.load(args.data_dir + '/torch_states_rtn_cvx.pth')),0)

torch_actions = torch.cat((torch.load(args.data_dir + '/torch_actions_scp.pth'),torch.load(args.data_dir + '/torch_actions_cvx.pth')),0)
torch_rtgs = torch.cat((torch.load(args.data_dir + '/torch_rtgs_scp.pth'),torch.load(args.data_dir + '/torch_rtgs_cvx.pth')),0)

n_data = torch_states.size(dim=0) # 数据组数量
n_time = torch_states.size(dim=1) # this excludes the target state time
n_state = torch_states.size(dim=2) # 状态量维度
n_action = torch_actions.size(dim=2) # 控制量维度
n_reward = 1

n_constraint = 1
torch_ctgs = torch.cat((torch.load(args.data_dir + '/torch_ctgs_scp.pth'),torch.load(args.data_dir + '/torch_ctgs_cvx.pth')),0)
n_mdp = n_state+n_action+n_reward+n_constraint # 马尔科夫决策过程维度

print('Completed\n')

# Normalize data
# 数据标准化，能够提高神经网络的性能和收敛速度
states_mean = torch_states.mean(dim=0)
states_std = (torch_states.std(dim=0) + 1e-6)

actions_mean = torch_actions.mean(dim=0)
actions_std = (torch_actions.std(dim=0) + 1e-6)

rtgs_mean = torch_rtgs.mean(dim=0)
rtgs_std = (torch_rtgs.std(dim=0) + 1e-6)

ctgs_mean = torch_ctgs.mean(dim=0)
ctgs_std = (torch_ctgs.std(dim=0) + 1e-6)

states_norm = ((torch_states - states_mean) / (states_std + 1e-6))
actions_norm = ((torch_actions - actions_mean) / (actions_std + 1e-6))
rtgs_norm = ((torch_rtgs - rtgs_mean) / (rtgs_std + 1e-6))
ctgs_norm = ((torch_ctgs - ctgs_mean) / (ctgs_std + 1e-6))

# Separate dataset in train and val data
n = int(0.9*n_data)
train_data = {'states':states_norm[:n, :], 'actions':actions_norm[:n, :], 'rtgs':rtgs_norm[:n, :], 'ctgs':ctgs_norm[:n, :]}
val_data = {'states':states_norm[n:, :], 'actions':actions_norm[n:, :], 'rtgs':rtgs_norm[n:, :], 'ctgs':ctgs_norm[n:, :]}

# RPDO data class
class RpodDataset(Dataset):
    # Create a Dataset object
    def __init__(self, split):
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.data = train_data if (self.split == 'train') or (self.split == 'val') else val_data
        self.n_data = len(self.data['states'])
        self.max_len = self.data['states'].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(self.n_data, (1,))                                   # 随机选一个数据集中的序号
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, n_state).float()       # 选中的数据集的轨迹状态量
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, n_action).float()      # 选中的数据集的轨迹控制量
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()             # 选中的数据集的轨迹reward-to-go值
        ctgs = torch.stack([self.data['ctgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()             # 选中的数据集的轨迹constraint-to-go值
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()
        return states, actions, rtgs, ctgs, timesteps, attention_mask, ix

    def get_data_size(self):
        return self.n_data

# Initialize dataset objects
train_dataset = RpodDataset('train')
print(len(train_dataset))
test_dataset = RpodDataset('val')
states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = train_dataset[0]

if verbose:
    print("states:", states_i.shape)
    print("actions:", actions_i.shape)
    print("rtgs:", rtgs_i.shape)
    print("ctgs:", rtgs_i.shape)
    print("timesteps:", timesteps_i.shape)
    print("attention_mask:", attention_mask_i.shape)

# create a DataLoader object for both train and test
# 训练中使用的批量大小为4，数据通过 PyTorch 的 DataLoader 加载
train_loader = DataLoader(
    train_dataset,
    sampler=torch.utils.data.RandomSampler(
        train_dataset, replacement=True, num_samples=int(1e10)),
    shuffle=False,
    pin_memory=True,
    batch_size=4,
    num_workers=0,
)
print(len(train_loader))
eval_loader = DataLoader(
    test_dataset,
    sampler=torch.utils.data.RandomSampler(
        test_dataset, replacement=True, num_samples=int(1e10)),
    shuffle=False,
    pin_memory=True,
    batch_size=4,
    num_workers=0,
)

config = DecisionTransformerConfig(
    state_dim=n_state,
    act_dim=n_action,
    hidden_size=384,    # 隐藏层大小
    max_ep_len=100,
    vocab_size=1,
    action_tanh=False,
    n_positions=1024,
    n_layer=6,          # 6层
    n_head=6,           # 6个注意力头
    n_inner=None,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    )

print('Intializing Transformer Model\n')
model = AutonomousRendezvousTransformer(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")
model.to(device);

# 使用的优化器是 AdamW，它是带有权重衰减正则化的 Adam 变体

optimizer = AdamW(model.parameters(), lr=3e-5) # 学习率3e-5
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

# for now this is unused. Potentially we can implement learning rate schedules
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)

# 可选的学习率调度器（get_scheduler）用于控制训练过程中的学习率。脚本指定了一个线性衰减的学习率调度
if use_lr_scheduler:
    num_training_steps = 10000000000
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps,
    )

# Eval function to plot results during training
eval_iters = 100
@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    losses_state = []
    losses_action = []
    for step in range(eval_iters):
        data_iter = iter(eval_dataloader)
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = next(data_iter)
        with torch.no_grad():
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
        loss_i = torch.mean((action_preds - actions_i) ** 2)
        loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
        losses.append(accelerator.gather(loss_i + loss_i_state))
        losses_state.append(accelerator .gather(loss_i_state))
        losses_action.append(accelerator.gather(loss_i))
    loss = torch.mean(torch.tensor(losses))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss_action = torch.mean(torch.tensor(losses_action))
    model.train()
    return loss.item(), loss_state.item(), loss_action.item()

print('\n======================')
print('Initializing training\n')
# Training loop
eval_steps = 500
samples_per_step = accelerator.state.num_processes * train_loader.batch_size

model.train()
completed_steps = 0
# 训练循环迭代整个数据集，进行指定次数的 epoch。在每次迭代中，模型预测的结果（状态和动作）与实际数据进行比较，并计算损失，损失是通过均方误差（MSE）计算的。
# 根据每4组轨迹数据，对数据集进行拟合，模型当中state[k]是t_{k+1}的预测状态，action[k]是t_{k}的预测控制
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        with accelerator.accumulate(model):
            states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = batch
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )

            loss_i_action = torch.mean((action_preds - actions_i) ** 2) # 第i次的控制量损失
            loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2) # 第i次的状态量损失

            # loss_i_target = torch.mean((state_preds[:,-1,:] - target_state) ** 2)
            loss = loss_i_action + loss_i_state # 第i次的总损失
            if step % 100 == 0:
                accelerator.print(
                    {
                        #"lr": lr_scheduler.get_lr(),
                        "lr":  lr_scheduler.get_lr() if use_lr_scheduler else optimizer.param_groups[0]['lr'],
                        "samples": step * samples_per_step,
                        "steps": completed_steps,
                        "loss/train": loss.item(),
                    }
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if use_lr_scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            if (step % (eval_steps)) == 0:
                eval_loss, loss_state, loss_action = evaluate()
                accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_action})
                model.train()
                accelerator.wait_for_everyone()
            if (step % eval_steps*10) == 0:
               accelerator.save_state(root_folder + '/transformer/saved_files/checkpoints/checkpoint_rtn_art')