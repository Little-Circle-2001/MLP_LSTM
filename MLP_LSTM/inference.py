# This Python code is used to apply trained MLP-LSTM Network to inference control action by past observe and control values

import os
import sys

import argparse



from torch.utils.data import DataLoader
from torch.utils.data import Dataset

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
data_dir = root_folder + '/dataset'

parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir

import torch
from MLP_LSTM.MLN import MLP_LSTM_Network
from optimization.ocp import *

verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# 数据标准化
data_stats = np.load(data_dir + '/dataset-rpod-stats.npz')
states_mean = data_stats['states_mean']
states_mean = torch.from_numpy(states_mean)
states_std = data_stats['states_std']
states_std = torch.from_numpy(states_std)
delta_states_target_mean = data_stats['delta_states_target_mean']
delta_states_target_mean = torch.from_numpy(delta_states_target_mean)
delta_states_target_std = data_stats['delta_states_target_std']
delta_states_target_std = torch.from_numpy(delta_states_target_std)
obstacle_mean = data_stats['obstacle_mean']
obstacle_mean = torch.from_numpy(obstacle_mean)
obstacle_std = data_stats['obstacle_std']
obstacle_std = torch.from_numpy(obstacle_std)
actions_mean = data_stats['actions_mean']
actions_mean = torch.from_numpy(actions_mean)
actions_std = data_stats['actions_std']
actions_std = torch.from_numpy(actions_std)


state_representation = 'rtn'

class MLP_LSTM_test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_size = self.data['observes'].shape[0]
        self.seq_len = self.data['observes'].shape[1]
        self.observe_dim = self.data['observes'].shape[2]
        self.action_dim = self.data['actions'].shape[2]
        self.obstacle_dim = self.data['obstacles'].shape[-1]
        self.max_num_obstacle = self.data['obstacles'].shape[2]

    def __len__(self):
        return self.data['observes'].shape[0]
        # return len(self.data)

    def __getitem__(self, idx):
        ix = idx
        # ix = torch.randint(self.data_size, (1,))
        observes = self.data['observes'][ix, :, :].view(self.seq_len, self.observe_dim)
        actions = self.data['actions'][ix, :, :].view(self.seq_len, self.action_dim)
        obstacles = self.data['obstacles'][ix, :, :, :].view(self.seq_len, self.max_num_obstacle, self.obstacle_dim)
        mask = self.data['masks'][ix, :, :].view(self.seq_len, self.max_num_obstacle)
        timesteps = torch.tensor(range(self.seq_len)).view(self.seq_len, 1)

        horizons = torch.tensor(self.data['data_param']['horizons'][ix].item())
        oe = torch.tensor(np.transpose(self.data['data_param']['oe'][ix])).unsqueeze(0)
        time_discr = torch.tensor(self.data['data_param']['time_discr'][ix].item())
        time_sec = torch.tensor(self.data['data_param']['time_sec'][ix].reshape((1, self.seq_len))).unsqueeze(0)

        return observes, actions, obstacles, mask, timesteps, oe, time_discr, time_sec, horizons, ix
def get_test_data():
    print("Loading data...")

    data_param = np.load(data_dir + '/dataset-rpod-v05-param.npz')

    data_param = {
        'horizons': np.concatenate((data_param['horizons'], data_param['horizons']), axis=0),
        'time_discr': np.concatenate((data_param['dtime'], data_param['dtime']), axis=0),
        'time_sec': np.concatenate((data_param['time'], data_param['time']), axis=0),
        'oe': np.concatenate((data_param['oe'], data_param['oe']), axis=0)
    }

    if state_representation == 'roe':
        torch_observe = torch.cat((torch.load(args.data_dir + '/torch_states_roe_scp.pth', weights_only=True), torch.load(args.data_dir + '/torch_states_roe_cvx.pth', weights_only=True)), dim=0)
        torch_delta_observe_target = torch.cat((torch.load(args.data_dir + '/torch_delta_state_roe_cvx.pth', weights_only=True),torch.load(args.data_dir + '/torch_delta_state_roe_scp.pth', weights_only=True)), dim=0)
    else:
        torch_observe = torch.cat((torch.load(args.data_dir + '/torch_states_rtn_scp.pth', weights_only=True), torch.load(args.data_dir + '/torch_states_rtn_cvx.pth', weights_only=True)), dim=0)
        torch_delta_observe_target = torch.cat((torch.load(args.data_dir + '/torch_delta_state_rtn_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_delta_state_rtn_scp.pth', weights_only=True)), dim=0)

    torch_obstacle = torch.cat((torch.load(args.data_dir + '/torch_delta_state_rtn_cvx_obs.pth', weights_only=True), torch.load(args.data_dir + '/torch_delta_state_rtn_scp_obs.pth', weights_only=True)), dim=0)
    torch_koz_vector = torch.cat((torch.load(args.data_dir + '/torch_koz_vector_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_koz_vector_scp.pth', weights_only=True)), dim=0)
    torch_action = torch.cat((torch.load(args.data_dir + '/torch_actions_cvx.pth', weights_only=True), torch.load(args.data_dir + '/torch_actions_scp.pth', weights_only=True)), dim=0)
    torch_mask = torch.cat((torch.load(args.data_dir + '/torch_mask.pth', weights_only=True), torch.load(args.data_dir + '/torch_mask.pth', weights_only=True)), dim=0)

    data_size = torch_observe.shape[0]
    
    print('Completed\n')

    observe_norm = (torch_observe - states_mean) / (states_std + 1e-6)
    delta_observe_target_norm = (torch_delta_observe_target - delta_states_target_mean) / (delta_states_target_std + 1e-6)
    delta_obstacle_norm = (torch_obstacle - obstacle_mean) / (obstacle_std + 1e-6)
    obstacle_norm = torch.cat((delta_obstacle_norm, torch_koz_vector), dim=-1)
    action_norm = (torch_action - actions_mean) / (actions_std + 1e-6)

    # Separate dataset in train and val data
    n_eval = int(0.9 * data_size)

    test_data = {
        'observes': observe_norm[n_eval:, :, :],
        'observes_target': delta_observe_target_norm[n_eval:, :, :],
        'obstacles': obstacle_norm[n_eval:, :, :, :],
        'masks': torch_mask[n_eval:, :, :],
        'actions': action_norm[n_eval:, :, :],
        'data_param': {
            'horizons': data_param['horizons'][n_eval:],
            'time_discr': data_param['time_discr'][n_eval:],
            'time_sec': data_param['time_sec'][n_eval:, :],
            'oe': data_param['oe'][n_eval:, :]
        },
        'data_stats': data_stats
    }
    test_dataset = MLP_LSTM_test_Dataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=64,
        num_workers=0,
        drop_last=False
    )
    return test_dataset, test_loader

def get_MLP_LSTM_model(network_model_name):
    class Config:
        def __init__(self):
            self.observe_dim = 6  # 输入特征的维度
            self.obstacle_dim = 4  # 障碍的维度
            self.max_obstacle = 5  # 最大允许观测到的障碍数量
            self.MLP_hidden_size = 128  # MLP 中间层的大小
            self.MLP_output_size = 5  # MLP 输出层的大小
            self.hidden_size = 128  # LSTM 隐藏层的大小
            self.num_layers = 2  # LSTM 层数
            self.action_dim = 9  # 输出的维度
            self.control_dim = 3
            self.seq_len = 40  # 输入序列的长度
            self.action_tanh = False

    config = Config()

    model = MLP_LSTM_Network(config)
    model_size = sum(t.numel() for t in model.parameters())
    # print(f"GPT size: {model_size / 1000 ** 2:.8f}M parameters")
    model.to(device)


    model.load_state_dict(torch.load(root_folder + '/MLP_LSTM/saved_files/checkpoints/' + network_model_name + '.pth', weights_only=False))
    return model.eval()


def model_inference_dyn(model, observe0, target, obstacle, action0, stm, cim, psi, state_representation, seq_len):
    if not isinstance(observe0, torch.Tensor):
        observe0 = torch.as_tensor(observe0)
    if not isinstance(action0, torch.Tensor):
        action0 = torch.as_tensor(action0)
    if not isinstance(target, torch.Tensor):
        target = torch.as_tensor(target)
    if not isinstance(obstacle, torch.Tensor):
        obstacle = torch.as_tensor(obstacle)

    state_dim = observe0.shape[0]
    action_dim = action0.shape[0]
    obstacle_dim = obstacle.shape[-1] + 1

    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)

    # 真实状态量与控制量（非标准化）
    roe_dyn = torch.empty(state_dim, seq_len)
    roe_trg_dyn = torch.empty(state_dim, seq_len)
    rtn_dyn = torch.empty(state_dim, seq_len)
    rtn_trg_dyn = torch.empty(state_dim, seq_len)
    obstacle_dyn = torch.empty(seq_len, max_num_obstacle, obstacle_dim)
    control_dyn = torch.empty(action_dim, seq_len)
    mask_dyn = torch.empty(seq_len, max_num_obstacle)
    time_dyn = torch.empty(seq_len, 1)

    # 神经网络内部控制量与状态量
    config = model.config

    # 全程变化的神经网络内部控制量与状态量（标准化后的）
    state_dyn = torch.empty(1, seq_len, state_dim)                                                  # [batch_size=1, seq_len, state_dim]
    state_trg_dyn = torch.empty(1, seq_len, state_dim)                                              # [batch_size=1, seq_len, state_dim]
    state_obstacle_dyn = torch.empty(1, seq_len, max_num_obstacle, obstacle_dim)                    # [batch_size=1, seq_len, max_num_obstacle, obstacle_dim]
    action_dyn = torch.empty(1, seq_len, action_dim)
    obstacle_mask = torch.empty(1, seq_len, max_num_obstacle)                                       # [batch_size=1, seq_len, action_dim]
    time_seq = torch.tensor(range(1 - config.seq_len, seq_len)).view(seq_len + config.seq_len - 1, 1)
    timestep_dyn = torch.empty(1, seq_len + config.seq_len - 1, 1)


    # 用于输入或输出的神经网络的内部状态量和控制量
    observe_inputs = torch.empty(1, config.seq_len, state_dim)                                      # [batch_size=1, seq_len, state_dim]
    observe_trg_inputs = torch.empty(1, config.seq_len, state_dim)                                  # [batch_size=1, seq_len, state_dim]
    observe_obstacle_inputs = torch.empty(1, config.seq_len, max_num_obstacle, obstacle_dim)        # [batch_size=1, seq_len, max_num_obstacle, obstacle_dim]
    action_inputs = torch.empty(1, config.seq_len, action_dim)                                      # [batch_size=1, seq_len, action_dim]
    mask_inputs = torch.empty(1, config.seq_len, max_num_obstacle)
    time_inputs = torch.empty(1, config.seq_len, 1)

    runtime0_MLP_LSTM = time.time()

    roe_dyn[:, 0] = observe0
    rtn_dyn[:, 0] = psi[:, :, 0] @ roe_dyn[:, 0]
    rtn_trg_dyn[:, 0] = target - rtn_dyn[:, 0]
    roe_trg_dyn[:, 0] = torch.from_numpy(map_rtn_to_roe(target, oe_0_ref)) - roe_dyn[:, 0]
    obstacle_dyn[0, :, :3] = obstacle - rtn_dyn[:3, 0].unsqueeze(0).tile(max_num_obstacle,1)
    obstacle_dyn[0, :, 3] = torch_check_koz_constraint(obstacle_dyn[0, :, :3], 0, max_num_obstacle)
    mask_dyn[0, :] = torch.from_numpy(np.array([1, 0, 0, 0, 0]))

    if state_representation == 'roe':
        state_dyn[:, 0, :] = (roe_dyn[:, 0] - states_mean[0]) / (states_std[0] + 1e-6)
        state_trg_dyn[:, 0, :] = (roe_trg_dyn[:, 0] - delta_states_target_mean[0]) / (delta_states_target_std[0] + 1e-6)
    elif state_representation == 'rtn':
        state_dyn[:, 0, :] = (rtn_dyn[:, 0] - states_mean[0]) / (states_std[0] + 1e-6)
        state_trg_dyn[:, 0, :] = (rtn_trg_dyn[:, 0] - delta_states_target_mean[0]) / (delta_states_target_std[0] + 1e-6)
    state_obstacle_dyn[:, 0, :, :] = torch.cat(((obstacle_dyn[0, :, :3] - obstacle_mean[0]) / (obstacle_std[0] + 1e-6), obstacle_dyn[0, :, 3].unsqueeze(-1)), dim=-1)
    obstacle_mask[:, 0, :] = mask_dyn[0, :]
    timestep_dyn[:, :, :] = (seq_len - 1 - time_seq) / (seq_len - 1)

    observe_inputs[:, :, :] = state_dyn[:, 0, :]
    observe_trg_inputs[:, :, :] = state_trg_dyn[:, 0, :]
    observe_obstacle_inputs[:, :, :, :] = state_obstacle_dyn[:, 0, :, :]
    mask_inputs[:, :, :] = obstacle_mask[:, 0, :]
    action_inputs[:, :, :] = action0
    time_inputs[:, :, :] = timestep_dyn[:, :config.seq_len, :]


    for t in range(seq_len):
        with torch.no_grad():
            outputs = model(
                observe=observe_inputs,
                delta_observe_target=observe_trg_inputs,
                obstacle=observe_obstacle_inputs,
                mask=mask_inputs,
                action=action_inputs,
                time=time_inputs,
                timestep=t + 1
            )
            output = outputs[:, min(config.seq_len-1, t), :3]                                 # [batch_size=1, action_dim]
        action_dyn[:, t, :] = output                                    # [batch_size=1, action_dim]

        action_inputs[:, max(-config.seq_len, -(t+1)):] = action_dyn[:, max(0, t - config.seq_len + 1):(t + 1), :]

        control_dyn[:, t] = action_dyn[:, t, :] * (actions_std[t] + 1e-6) + actions_mean[t]

        if t != seq_len - 1:
            roe_dyn[:, t + 1] = stm[:, :, t] @ (roe_dyn[:, t] + cim[:, :, t] @ control_dyn[:, t])
            rtn_dyn[:, t + 1] = psi[:, :, t + 1] @ roe_dyn[:, t + 1]
            roe_trg_dyn[:, t + 1] = torch.from_numpy(map_rtn_to_roe(target, oe_0_ref)) - roe_dyn[:, t + 1]
            rtn_trg_dyn[:, t + 1] = target - rtn_dyn[:, t + 1]
            if state_representation == 'roe':
                state_dyn[:, t + 1, :] = (roe_dyn[:, t + 1] - states_mean[t + 1]) / (states_std[t + 1] + 1e-6)
                state_trg_dyn[:, t + 1, :] = (roe_trg_dyn[:, t + 1] - delta_states_target_mean[t + 1]) / (delta_states_target_std[t + 1] + 1e-6)
            elif state_representation == 'rtn':
                state_dyn[:, t + 1, :] = (rtn_dyn[:, t + 1] - states_mean[t + 1]) / (states_std[t + 1] + 1e-6)
                state_trg_dyn[:, t + 1, :] = (rtn_trg_dyn[:, t + 1] - delta_states_target_mean[t + 1]) / (delta_states_target_std[t + 1] + 1e-6)

            obstacle_dyn[t + 1, :, :3] = obstacle - rtn_dyn[:3, t + 1].unsqueeze(0).tile(max_num_obstacle, 1)
            obstacle_dyn[t + 1, :, 3] = torch_check_koz_constraint(obstacle_dyn[t + 1, :, :3], t + 1, max_num_obstacle)
            state_obstacle_dyn[:, t + 1, :, :] = torch.cat(((obstacle_dyn[t + 1, :, :3] - obstacle_mean[t + 1]) / (obstacle_std[t + 1] + 1e-6), obstacle_dyn[t + 1, :, 3].unsqueeze(-1)), dim=-1)

            mask_dyn[t + 1, :] = torch.from_numpy(np.array([1, 0, 0, 0, 0]))
            obstacle_mask[:, t + 1, :] = mask_dyn[t + 1, :]

            # timestep_dyn[:, t + 1, :] = (seq_len - 1 - (t + 1)) / (seq_len - 1)
            # timestep_dyn[t, :] = (seq_len - (t + 1)) / seq_len

            observe_inputs[:, max(-config.seq_len, -(t + 2)):, :] = state_dyn[:, max(0, t - config.seq_len + 2):(t + 2),:]                              # [batch_size=1, seq_len, state_dim]
            observe_trg_inputs[:, max(-config.seq_len, -(t + 2)):, :] = state_trg_dyn[:, max(0, t - config.seq_len + 2):(t + 2),:]                      # [batch_size=1, seq_len, state_dim]
            observe_obstacle_inputs[:, max(-config.seq_len, -(t + 2)):, :, :] = state_obstacle_dyn[:, max(0, t - config.seq_len + 2):(t + 2), :, :]     # [batch_size=1, seq_len, max_num_obstacle, obstacle_dim]
            mask_inputs[:, max(-config.seq_len, -(t + 2)):, :] = obstacle_mask[:, max(0, t - config.seq_len + 2):(t + 2), :]                            # [batch_size=1, seq_len, max_num_obstacle]
            time_inputs[:, :, :] = timestep_dyn[:, (t + 1):(t + 1 + config.seq_len), :]

    runtime1_MLP_LSTM = time.time()
    runtime_MLP_LSTM = runtime1_MLP_LSTM - runtime0_MLP_LSTM

    MLP_LSTM_Trajectory = {
        'roe_dyn': roe_dyn.numpy(),
        'rtn_dyn': rtn_dyn.numpy(),
        'control_dyn': control_dyn.numpy()
    }

    return MLP_LSTM_Trajectory, runtime_MLP_LSTM


def torch_check_koz_constraint(states_rtn, n_time, max_num_obstacle):

    # Ellipse equation check for a single instant in the trajectory
    constr_koz_violation = torch.empty(max_num_obstacle,)
    constr_koz = torch.empty(max_num_obstacle, 1)

    for i in range(max_num_obstacle):
        constr_koz[i, :] = states_rtn[i, :] @ (torch.from_numpy(EE_koz).float().to(device) @ states_rtn[i, :])
        if (constr_koz[i, :] < 1) and (n_time < dock_wyp_sample):
            constr_koz_violation[i] = 1.
        else:
            constr_koz_violation[i] = 0.

    return constr_koz_violation



