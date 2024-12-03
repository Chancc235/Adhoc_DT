import pickle
import torch

# 文件路径
file_path = './saves/overcooked/overcooked_trajectorys1/buffer_5120.pkl'
# 打开并读取 .plk 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)
reward = data["transition_data"]["reward"].squeeze(-1)[:1024]  # 变为 [5000, 201]
ter = data["transition_data"]["terminated"].squeeze(-1)[:1024]
print(torch.sum(ter == 1).item())

# 对每个 episode 的奖励求和，得到每个 episode 的总回报
episode_returns = reward.sum(dim=1)  # 结果形状为 [5000]

print(episode_returns.mean())

print(data["transition_data"]["state"].shape)
print(data["transition_data"]["obs"].shape)
print(data["transition_data"]["actions"].shape)
print(data["transition_data"]["avail_actions"].shape)
print(data["transition_data"]["reward"].shape)
print(data["transition_data"]["terminated"].shape)

'''
for i in data["transition_data"]["terminated"]:
    if any(i):
        print(i)



tensor = data["transition_data"]["reward"]

# 统计等于 1 的元素个数
count = torch.sum(tensor).item()
print(count)
file_path = './saves/buffer_2048.pkl'
# 打开并读取 .plk 文件
with open(file_path, 'rb') as file:
    data2 = pickle.load(file)

print(data2["transition_data"]["state"].shape)
print(data2["transition_data"]["obs"].shape)
print(data2["transition_data"]["actions"].shape)
print(data2["transition_data"]["avail_actions"].shape)
print(data2["transition_data"]["reward"].shape)
print(data2["transition_data"]["terminated"].shape)
'''