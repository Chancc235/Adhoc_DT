import torch
from Networks.TeammateEncoder import TeammateEncoder
from Networks.AdhocAgentEncoder import AdhocAgentEncoder
from Networks.ReturnNet import ReturnNet
from Networks.GoalDecoder import GoalDecoder

# 假设有 10 个 agent，每个 agent 的 state_dim 是 16
num_agents = 4
state_dim = 75
embed_dim = 32
num_heads = 4
batch_size = 1


# 队友编码器
encoder = TeammateEncoder(state_dim, embed_dim, num_heads)
states = torch.randn(num_agents, batch_size, state_dim)  # (num_agents, batch_size, state_dim)
print(states.shape)
encoded_output, _ = encoder(states)  # (batch_size, embed_dim)
print(encoded_output.shape)  # 输出应该是 (batch_size, embed_dim)
print(encoded_output)


# 个体编码器
encoder2 = AdhocAgentEncoder(state_dim, embed_dim)
state = torch.randn(batch_size, state_dim)  # (batch_size, state_dim)

encoder2.eval()
encoded_output2, _ = encoder2(state)  # (batch_size, embed_dim)
print(encoded_output2.shape)  # 输出应该是 (batch_size, embed_dim)
print(encoded_output2)


# return net
model = ReturnNet(embed_dim, 256)
model.eval()
returns = model(encoded_output)
print(returns.shape)
print(returns)

# 目标解码器
goal_decoder = GoalDecoder(embed_dim, 1, 512, state_dim, 4)
goal_decoder.eval()
goal = goal_decoder(encoded_output, returns)
goal = (goal > 0.5).float()

print(goal.shape)
print(goal)