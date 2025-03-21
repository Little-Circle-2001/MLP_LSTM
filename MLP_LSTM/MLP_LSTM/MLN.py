import torch
from torch import nn


class MLP_LSTM_Network(nn.Module):
    def __init__(self, config):
        super(MLP_LSTM_Network, self).__init__()
        self.hidden_size = config.hidden_size
        self.action_dim = config.action_dim
        self.MLP_out = nn.Sequential(
            nn.Linear(in_features=config.observe_dim+config.obstacle_dim*config.max_obstacle, out_features=config.MLP_hidden_size),
            nn.Linear(in_features=config.MLP_hidden_size, out_features=config.MLP_output_size),
            nn.ReLU()
        )

        self.LSTM = nn.LSTM(input_size=config.MLP_output_size + config.action_dim, hidden_size=config.hidden_size,
                            num_layers=config.num_layers, batch_first=True)


        self.LSTM_out = nn.Linear(in_features=config.hidden_size, out_features=config.action_dim)


        # self.lstm = nn.LSTM(input_size=config.input_size+config.action_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True)
        # self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.action_dim)

    def forward(self, observe, obstacle, mask):
        batch_size, seq_len, observe_dim = observe.shape
        _, _, max_obstacle_num, obstacle_dim = obstacle.size()

        multi_input = torch.cat((observe, (obstacle*mask.unsqueeze(-1)).view(batch_size, seq_len, -1)), dim=-1)
        # [batch_size, seq_len, observe_dim+max_obstacle*obstacle_dim]

        MLP_out = []

        for t in range(seq_len):
            x = multi_input[:, t, :]            # [batch_size, observe_dim+max_obstacle*obstacle_dim]
            x = self.MLP_out(x)                 # [batch_size, MLP_action_dim]
            MLP_out.append(x.unsqueeze(1))      # [batch_size, 1, MLP_action_dim]
        MLP_out = torch.cat(MLP_out, dim=1)     # [batch_size, seq_len, MLP_action_dim]

        h0 = torch.zeros(1, batch_size, self.hidden_size)           # [batch_size, hidden_size]
        c0 = torch.zeros(1, batch_size, self.hidden_size)           # [batch_size, hidden_size]
        output0 = torch.zeros(batch_size, self.action_dim)              # [batch_size, action_dim]
        output = torch.zeros(batch_size, seq_len, self.action_dim)
        for t in range(seq_len):
            LSTM_input = torch.cat((MLP_out[:, t, :], output0), dim=-1)     # [batch_size, MLP_action_dim+action_dim]
            LSTM_output, (hn, cn) = self.LSTM(LSTM_input.unsqueeze(1), (h0, c0))    # [batch_size, 1, hidden_size]
            o = self.LSTM_out(LSTM_output)                                          # [batch_size, 1, action_dim]
            h0 = hn
            c0 = cn
            output0 = o.squeeze(1)
            output[:, t, :] = output0                                               # [batch_size, 1, action_dim]
        return output


# 配置文件类，传入超参数
class Config:
    def __init__(self):
        self.observe_dim = 7       # 输入特征的维度
        self.obstacle_dim = 3      # 障碍的维度
        self.max_obstacle = 4     # 最大允许观测到的障碍数量
        self.MLP_hidden_size = 128  # MLP 中间层的大小
        self.MLP_output_size = 2
        self.hidden_size = 64       # LSTM 隐藏层的大小
        self.num_layers = 1         # LSTM 层数
        self.action_dim = 3        # 输出的维度
        self.seq_len = 20           # 输入序列的长度
        self.action_tanh = False

# 创建config实例
config = Config()

# 构建模型
model = MLP_LSTM_Network(config)

# 示例输入数据，batch_size=32, seq_len=20, input_size=10
observe = torch.randn(32, 100, config.observe_dim)                              # [batch_size, seq_len, observe_dim]

obstacle = torch.randn(32, 100, config.max_obstacle, config.obstacle_dim)       # [batch_size, seq_len, max_obstacles, obstacle_dim]
mask = torch.randint(0, 2, (32, 100, config.max_obstacle)).float()               # [batch_size, seq_len, max_obstacles]，1表示有效，0表示无效


# 前向传播
output = model(observe, obstacle, mask)

# 输出预测结果
print(output)  # 输出的形状应为 [batch_size, seq_len, action_dim]