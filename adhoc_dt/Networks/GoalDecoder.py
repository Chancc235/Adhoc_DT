import torch
import torch.nn as nn

class GoalDecoder(nn.Module):
    def __init__(self, input_dim, scalar_dim, hidden_dim, output_dim):
        super(GoalDecoder, self).__init__()
        
        # 先将输入向量和数值拼接
        self.fc1 = nn.Linear(input_dim + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.01)
        self.out_activation = nn.Sigmoid()

    def forward(self, vector_input, scalar_input):
        
        # 将 vector_input 和 scalar_input 在最后一个维度上拼接
        x = torch.cat((vector_input, scalar_input), dim=1)  # (batch_size, input_dim + 1)
        
        x = self.fc1(x)
        x = self.activation(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        
        x = self.fc3(x)
        x = self.out_activation(x)
        
        return x