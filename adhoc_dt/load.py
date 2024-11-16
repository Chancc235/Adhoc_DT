import torch
data_path = "data/PP4a_test.pt"
episodes_data = torch.load(data_path)
print(len(episodes_data))
print(episodes_data[0].keys())
print(episodes_data[0]["obs"][0])
print(episodes_data[0]["state"].shape)


print(episodes_data[0]["teammate_action"].shape)
