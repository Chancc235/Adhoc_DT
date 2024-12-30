import torch

data = torch.load("./data/PP4a_episodes_datas_rtg_new.pt")
#print(data[1]["rtg"].shape)
print(len(data))
print(data[0]["reward"].shape)
print(data[0]["action"].shape)
print(data[0]["obs"].shape)
print(data[0]["state"].shape)

