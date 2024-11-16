import torch
import torch.nn as nn

class GoalDecoder(nn.Module):
    def __init__(self, input_dim, scalar_dim, hidden_dim, output_dim, num):
        super(GoalDecoder, self).__init__()
        
        # 先将输入向量和数值拼接
        self.fc1 = nn.Linear(input_dim + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim * num)
        self.num = num
        # 激活函数
        self.activation = nn.LeakyReLU(0.01)
        self.out_activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)


    def forward(self, vector_input, scalar_input):
        
        # 将 vector_input 和 scalar_input 在最后一个维度上拼接
        x = torch.cat((vector_input, scalar_input), dim=1)  # (batch_size, input_dim + 1)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.out_activation(x)
        x = x.view(-1, self.num, 75)
        x = x.permute(1, 0, 2)  # (num_agents, batch_size, state_dim)

        # x = (x > 0.5).float()
        
        return x