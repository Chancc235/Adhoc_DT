import torch

data = torch.load("./data/overcooked_episodes_datas_rtg_new3.pt")
#print(data[1]["rtg"].shape)
print(len(data))
print(data[0]["reward"].shape)
print(data[0]["action"].shape)
print(data[0]["obs"].shape)
print(data[0]["state"].shape)
# print(data[100]["state"][:5])

for i in data:
    unique_labels = torch.unique(i["state"])
    
    if not torch.all(torch.isin(unique_labels, torch.tensor([0, 1]))):
        invalid_labels = unique_labels[~torch.isin(unique_labels, torch.tensor([0, 1]))]
        print(f"Invalid labels found: {invalid_labels}")
        tensor_str = str(i["state"].tolist())  # 使用 .tolist() 将 tensor 转换为列表，然后转换为字符串
        # 打开一个 txt 文件并写入
        with open("tensor_output.txt", "w") as f:
            f.write(tensor_str)
        break
    else:
        print(1)
'''
print(data[0]['action'])
print(data[0]['state'][:3])
print(data[0]['state'][3:6])
print(data[0]['state'][6:9])
'''


