import torch
import torch.nn.functional as F
from utils_dt import preprocess_data
import random
from tqdm import tqdm

def to_onehot(onehot_tensor, indices_to_onehot=[9, 10, 23, 24]):
    onehot_tensor = onehot_tensor[..., :29]
    onehot_tensor = torch.clamp(onehot_tensor, min=0)
    indices_to_onehot = [v + i * 9 for i, v in enumerate(indices_to_onehot)] # [9, 19, 41, 51]
    for index in indices_to_onehot:
        values = onehot_tensor[..., index].to(torch.int64)
        # 将该维度的值转为 one-hot 编码
        onehot_values = F.one_hot(values, num_classes=12) 
        onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

    return onehot_tensor.to(torch.float32)

data_path = "data/overcooked_episodes_datas_rtg_new.pt"
sav_data_path = "data/overcooked_episodes_datas_rtg_new2.pt"
data = torch.load(data_path)
new_data = []
print("loaded")

for i, v in enumerate(data):
    s = to_onehot(v['state'])
    n = to_onehot(v['next_state'])
    o = to_onehot(v['obs'])

    data_dict = {
        'state': s,
        'next_state': n,
        'obs': o,
        'action': v['action'],
        'reward': v['reward'],
        'done': v['done'],
        'teammate_action': v['teammate_action'],
        'rtg': v['rtg']
    }
    new_data.append(data_dict)
    print(f"{i} finished")
# print(new_data[-1]['state'].shape)
torch.save(new_data, sav_data_path)
print("end")