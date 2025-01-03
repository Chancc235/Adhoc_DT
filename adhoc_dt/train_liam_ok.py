import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import random
import csv
import time

from utils_dt import load_config, create_save_directory, setup_logger, save_config, get_batch_dt, preprocess_data
from Networks.LiamModule import ValueNetwork, QNetwork, PolicyNetwork, ReconstructionDecoder, Encoder
from Data import CustomDataset
from Trainer import LiamTrainer
from TestGame import Test
from Agent.LiamAgent import LiamAgent

# 定义训练函数
def train_model(logger, trainer, train_loader, val_loader, num_epochs, device, test_interval, save_interval, save_dir, act_dim, model_save_path="models"):
    start_time = time.time()
    # 测试类
    test = Test("overcooked")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, episodes_data in enumerate(pbar):

                loss_dict = trainer.train(episodes_data, max_ep_len=episodes_data["state"].size(1))

                epoch_loss += loss_dict['total_loss']
                # 打印每个batch的损失
                pbar.set_postfix({
                    "Total Loss": f"{loss_dict['total_loss']:.4f}"
                })

                # 日志记录
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}]\n "
                                f"Total Loss: {loss_dict['total_loss']:.4f}\n"
                                f"a2c Loss: {loss_dict['a2c_loss']:.4f}\n"
                                f"reconstruction Loss: {loss_dict['reconstruction_loss']:.4f}\n ")
            
        # 每个epoch结束后记录平均损失
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Total Loss: {epoch_loss / len(train_loader) :.4f}")
        logger.info("===================================================================================")


        # 将损失写入 CSV 文件
        val_csv_file_path = os.path.join(save_dir, 'val_loss.csv')
        with open(val_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow([epoch + 1, dt_val_loss / len(val_loader), epoch_loss / len(train_loader)])
            writer.writerow([epoch + 1, epoch_loss / len(train_loader)])
        # 计算训练时间
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours, rem = divmod(epoch_duration, 3600)
        minutes, _ = divmod(rem, 60)
        logger.info(f"Completed in {int(hours)}h {int(minutes)}m")
        
        # 每隔指定的间隔进行测试
        if (epoch + 1) % test_interval == 0 or epoch + 1 == 1:
            agent = LiamAgent(trainer.liam_encoder , trainer.policy_net, "overcooked")
            returns, var = test.test_game_liam(50, agent)
            logger.info(f"{epoch + 1} Test Returns: {returns}")
            returns_csv_file_path = os.path.join(save_dir, 'test_returns.csv')
            with open(returns_csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, returns, var])
        
    end_time = time.time()
    total_duration = end_time - start_time
    total_hours, total_rem = divmod(total_duration, 3600)
    total_minutes, _ = divmod(total_rem, 60)
    logger.info(f"Training completed in {int(total_hours)}h {int(total_minutes)}m")

if __name__ == "__main__":

    env = "overcooked"
    config = load_config(f"./config/{env}_config_liam.yaml")
    save_dir = create_save_directory()
    config["save_dir"] = save_dir
    logger = setup_logger(save_dir)
    
    # 训练设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    device = config["device"]
    # 移动模型到设备
    encoder = Encoder(
        obs_dim=config["state_dim"],
        action_dim=config["act_dim"],
        z_dim=config["z_dim"]
    )
    reconstruction_decoder = ReconstructionDecoder(
        z_dim=config["z_dim"],
        obs_dim=config["state_dim"],
        action_dim=config["act_dim"],
        num_agents=config["num_agents"]
    )
    policy_net = PolicyNetwork(
        z_dim=config["z_dim"],
        action_dim=config["act_dim"],
        obs_dim=config["state_dim"]
    )
    Q_net = QNetwork(
        obs_dim=config["state_dim"],
        action_dim=config["act_dim"],
        z_dim=config["z_dim"]
    )
    V_net = ValueNetwork(
        obs_dim=config["state_dim"],
        z_dim=config["z_dim"]
    )
    
    # trainer准备
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(reconstruction_decoder.parameters()) +
        list(Q_net.parameters()) +
        list(V_net.parameters()) +
        list(policy_net.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 初始化 Trainer
    trainer = LiamTrainer(
        liam_encoder=encoder,
        reconstruction_decoder=reconstruction_decoder,
        Q_net=Q_net,
        V_net=V_net,
        policy_net=policy_net,
        optimizer=optimizer,  
        gamma=config["gamma"],
        beta=config["beta"],
        alpha=config["alpha"],
        sita=config["sita"],
        batch_size=config["batch_size"],
        act_dim=config["act_dim"],
        update_freq=config["update_freq"],
        device=config["device"]
    )

    # 保存配置到输出文件夹
    save_config(config, save_dir)
    logger.info("Starting.")
    logger.info("Loading Data.")
    data_path = config["train_data_path"]

    # 加载数据
    load_start_time = time.time()
    data = torch.load(data_path)
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    hours, rem = divmod(load_duration, 3600)
    minutes, _ = divmod(rem, 60)
    logger.info(f"Data loaded in {hours} hours {minutes} minutes.")

    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.7, random_state=42)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    logger.info("Training Started.")
    # 开始训练
    train_model(logger, trainer, 
                train_loader, val_loader,
                num_epochs=config["num_epochs"], 
                device=config["device"], 
                test_interval=config["test_interval"],
                save_interval=config["save_interval"], 
                save_dir=save_dir, 
                act_dim=config["act_dim"])
                 
    # test(train_loader, device=config["device"], num_epochs=config["num_epochs"], batch_size=config["batch_size"])
    logger.info("Training completed.")