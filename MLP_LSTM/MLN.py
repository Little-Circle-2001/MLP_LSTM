import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MLP_LSTM_Network(nn.Module):
    def __init__(self, config):
        super(MLP_LSTM_Network, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.action_dim = config.action_dim
        self.num_layers = config.num_layers
        self.seq_len = config.seq_len
        self.MLP_out = nn.Sequential(
            nn.Linear(
                in_features=2 * config.observe_dim + config.obstacle_dim * config.max_obstacle,
                out_features=config.MLP_hidden_size
            ),
            nn.Linear(
                in_features=config.MLP_hidden_size,
                out_features=config.MLP_hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.MLP_hidden_size,
                out_features=config.MLP_output_size
            ),
            nn.ReLU()
        )

        self.LSTM = nn.LSTM(
            input_size=config.MLP_output_size + config.control_dim + 1,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        self.LSTM_out = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.action_dim
        )

    # def forward(self, observe, obstacle, mask):
    def forward(self, observe, delta_observe_target, obstacle, mask, action, time, timestep):
        batch_size, seq_len, observe_dim = observe.shape

        multi_input = torch.cat((observe, delta_observe_target, (obstacle * mask.unsqueeze(-1)).view(batch_size, seq_len, -1)), dim=-1)
        # [batch_size, seq_len, observe_dim+max_obstacle*obstacle_dim]

        MLP_out = self.MLP_out(multi_input)       # [batch_size, seq_len, MLP_output_dim]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)           # [batch_size, hidden_size]
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)           # [batch_size, hidden_size]

        pad_MLP_out = MLP_out[:, max(-timestep, -self.seq_len):, :]
        pad_action_input = action[:, max(-timestep, -self.seq_len):, :]
        pad_time_input = time[:, max(-timestep, -self.seq_len):, :]
        input = torch.cat((pad_MLP_out, pad_action_input, pad_time_input), dim=-1)

        pad_input, length = self.pad_left(input)                            # [batch_size, seq_len, MLP_action_dim + action_dim + 1]
        LSTM_input = pack_padded_sequence(pad_input, lengths=length, batch_first=True, enforce_sorted=True)

        LSTM_out, (hn, cn)= self.LSTM(LSTM_input, (h0, c0))                 # [batch_size, sen_len, hidden_size]
        packed_out, _ = pad_packed_sequence(LSTM_out, batch_first=True, total_length=self.seq_len)

        output = self.LSTM_out(packed_out)                                  # [batch_size, seq_len, action_dim]

        return output

    def pad_left(self, x):
        # 输入的x维度为 [batch_size, actual_len, x_size]
        # length 为当前想要保留的有效长度
        batch_size, actual_len, x_size = x.shape
        if actual_len < self.seq_len:
            y = torch.cat((x[:, -actual_len:, :], torch.zeros(batch_size, self.seq_len - actual_len, x_size)), dim=1)
            length = torch.ones(batch_size) * actual_len
        else:
            y = x
            length = torch.ones(batch_size) * self.seq_len
        return y, length


# # 配置文件类，传入超参数
# class Config:
#     def __init__(self):
#         self.observe_dim = 6     # 输入特征的维度
#         self.obstacle_dim = 4     # 障碍的维度
#         self.max_obstacle = 5     # 最大允许观测到的障碍数量
#         self.MLP_hidden_size = 128  # MLP 中间层的大小
#         self.MLP_output_size = 5
#         self.hidden_size = 64       # LSTM 隐藏层的大小
#         self.num_layers = 1         # LSTM 层数
#         self.action_dim = 9        # 输出的维度
#         self.seq_len = 20           # 输入序列的长度
#         self.control_dim = 3
#         self.action_tanh = False
#
# # 创建config实例
# config = Config()
#
# # 构建模型
# model = MLP_LSTM_Network(config)
#
# # 示例输入数据，batch_size=32, seq_len=20, input_size=10
# observe = torch.randn(32, 30, config.observe_dim)                              # [batch_size, seq_len, observe_dim]
# delta_observe_target = torch.randn(32, 30, config.observe_dim)
# action = torch.randn(32, 30, config.control_dim)
#
# obstacle = torch.randn(32, 30, config.max_obstacle, config.obstacle_dim)       # [batch_size, seq_len, max_obstacles, obstacle_dim]
# mask = torch.randint(0, 2, (32, 30, config.max_obstacle)).float()              # [batch_size, seq_len, max_obstacles]，1表示有效，0表示无效
# time = torch.randn(32, 30, 1)
#
#
# # 前向传播
# output = model(
#     observe=observe,
#     delta_observe_target=delta_observe_target,
#     action=action,
#     obstacle=obstacle,
#     mask=mask,
#     time=time,
#     timestep=2
# )
#
# # 输出预测结果
# print(output.shape)  # 输出的形状应为 [batch_size, seq_len, action_dim]