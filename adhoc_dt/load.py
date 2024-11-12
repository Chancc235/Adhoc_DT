import torch
data_path = "data/PP4a_test.pt"
episodes_data = torch.load(data_path)
print(len(episodes_data))
print(episodes_data[0].keys())
print(episodes_data[0]["obs"][0])
print(episodes_data[0]["state"].shape)

for i in range(0, len(episodes_data)):
    a = episodes_data[i]["action"]
    print(a[a<0])
print(episodes_data[0]["teammate_action"].shape)
