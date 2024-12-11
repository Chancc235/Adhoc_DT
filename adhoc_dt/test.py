import torch
data = torch.load("./data/overcooked_episodes_datas_rtg_new2.pt")
#print(data[1]["rtg"].shape)
print(len(data))
print(data[0]["reward"].shape)
print(data[0]["action"].shape)
print(data[0]["obs"].shape)
print(data[0]["state"].shape)
# print(data[100]["state"][:5])



print(data[0]['action'])
print(data[0]['state'][:3])
print(data[0]['state'][3:6])
print(data[0]['state'][6:9])



