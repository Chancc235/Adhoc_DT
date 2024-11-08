import pickle
# 文件路径
file_path = './saves/buffer_10240.pkl'
txt_file_path = 'output.txt'
# 打开并读取 .plk 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 将数据写入 .txt 文件
with open(txt_file_path, 'w') as txt_file:
    # 判断数据类型并写入
    if isinstance(data, (list, tuple)):
        for item in data:
            txt_file.write(f"{item}\n")
    elif isinstance(data, dict):
        for key, value in data.items():
            txt_file.write(f"{key}: {value}\n")
    else:
        txt_file.write(f"{data}\n")  # 处理单个对象

print(f"数据已成功写入到 {txt_file_path}")