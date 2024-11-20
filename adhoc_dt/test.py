import torch
from utils_dt import preprocess_data
data_path = "data/PP4a_test.pt"
# sav_data_path = "data/PP4a_episodes_datas_rtg.pt"
# data_path = "data/PP4a_test.pt"
data = torch.load(data_path)
print(data[10]["rtg"])
print(data[10]["reward"])
tensor_list = data[10]["rtg"]
scalar_tensor = torch.tensor([t.item() for t in tensor_list])

print(scalar_tensor)
# data = preprocess_data(data)
# print(data[0]["rtg"])
# print(data[0]["reward"])
# torch.save(data, sav_data_path)