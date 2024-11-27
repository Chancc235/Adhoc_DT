import torch
data = torch.load("./data/LBF_episodes_datas_rtg.pt")
print(data[1]["rtg"].shape)
print(data[1]["action"].shape)
print(data[1]["obs"].shape)
print(data[1]["state"].shape)
print(data[1]["action"])
for o in data[1]["state"]:
    print(o)