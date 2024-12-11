import torch
from utils_dt import preprocess_data
import random
from tqdm import tqdm
data_path1 = "data/overcooked_episodes_datas_rtg.pt"
data_path2 = "data/overcooked_episodes_datas2_rtg.pt"
sav_data_path = "data/overcooked_episodes_datas_rtg_new.pt"
data1 = torch.load(data_path1)
data2 = torch.load(data_path2)
print("loaded")
list_data = []
for i in tqdm(data1):
    if i["rtg"][0] != 0:
        list_data.append(i)
    else:
        rn = random.randint(1, 2)
        if rn == 1:
            list_data.append(i)
for i in tqdm(data2):
    if i["rtg"][0] != 0:
        list_data.append(i)
    else:
        rn = random.randint(1, 2)
        if rn == 1:
            list_data.append(i)
print(len(list_data))
torch.save(list_data, sav_data_path)