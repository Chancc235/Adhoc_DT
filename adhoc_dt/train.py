import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import load_config, create_save_directory, setup_logger, save_config
import os

from Networks.ReturnNet import ReturnNet
from Networks.TeammateEncoder import TeammateEncoder
from Networks.AdhocAgentEncoder import AdhocAgentEncoder
from Networks.GoalDecoder import GoalDecoder

from Data import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Trainer import Trainer
import time

# 定义训练函数
def train_model(logger, trainer, train_loader, val_loader, num_epochs, device, test_interval, save_interval, save_dir, model_save_path="models"):
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, episodes_data in enumerate(pbar):

                states = episodes_data["state"]
                obs = episodes_data["obs"]
                reward = episodes_data["reward"]
                goal = episodes_data["next_state"]
                
                for ts in range(states.size(1)):
                    s = states[:, ts, :, :].permute(1, 0, 2).to(device)   # shape [batch, num, dim]
                    o = obs[:, ts, :].to(device)   # shape [batch, dim]
                    r_true = reward[:, ts].to(device)   # shape [batch, 1]
                    g_true = goal[:, ts, :, :].permute(1, 0, 2).to(device)  # shape [batch, num, dim]
                    
                    loss_dict = trainer.train_step(s, o, r_true, g_true)
                    epoch_loss += loss_dict["total_loss"]

                # 更新进度条和打印每个batch的损失
                pbar.set_postfix({
                    "Total Loss": f"{loss_dict['total_loss']:.4f}",
                    "MIE Loss": f"{loss_dict['mie_loss']:.4f}",
                    "MSE Loss R": f"{loss_dict['mse_loss_r']:.4f}",
                    "MSE Loss G": f"{loss_dict['mse_loss_g']:.4f}"
                })
                
                # 日志记录
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                                f"Total Loss: {loss_dict['total_loss']:.4f}, MIE Loss: {loss_dict['mie_loss']:.4f}, "
                                f"MSE Loss R: {loss_dict['mse_loss_r']:.4f}, MSE Loss G: {loss_dict['mse_loss_g']:.4f}")
        
        # 每个epoch结束后记录平均损失
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Total Loss: {epoch_loss / len(train_loader):.4f}")
        
        val_loss_dict = trainer.evaluate(val_loader, device)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss_dict['total_loss']:.4f}")
        
        # 计算训练时间
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours, rem = divmod(epoch_duration, 3600)
        minutes, _ = divmod(rem, 60)
        logger.info(f"Completed in {int(hours)}h {int(minutes)}m")

        # 每隔指定的间隔进行测试
        if (epoch + 1) % test_interval == 0:
            pass
        # 每隔指定的间隔保存模型
        if (epoch + 1) % save_interval == 0:
            dir_path = os.path.join(save_dir, model_save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_path = os.path.join(dir_path, f"epoch_{epoch+1}.pth") 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': {
                    'teamworkencoder': trainer.teammateencoder.state_dict(),
                    'adhocencoder': trainer.adhocencoder.state_dict(),
                    'returnnet': trainer.returnnet.state_dict(),
                    'goaldecoder': trainer.goaldecoder.state_dict(),
                },
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader)
            }, save_path)
            logger.info(f"Model checkpoint saved at {save_path}")
    end_time = time.time()
    total_duration = end_time - start_time
    total_hours, total_rem = divmod(total_duration, 3600)
    total_minutes, _ = divmod(total_rem, 60)
    print(f"Training completed in {int(total_hours)}h {int(total_minutes)}m")

if __name__ == "__main__":
    env = "PP4a"
    config = load_config(f"./config/{env}_config.yaml")
    save_dir = create_save_directory()
    config["save_dir"] = save_dir
    logger = setup_logger(save_dir)
    
    # 训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 移动模型到设备
    teammateencoder = TeammateEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"], num_heads=config["TeammateEncoder_num_heads"]).to(device)
    adhocagentEncoder = AdhocAgentEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"]).to(device)
    returnnet = ReturnNet(input_dim=config["embed_dim"]).to(device)
    goaldecoder = GoalDecoder(input_dim=config["embed_dim"], scalar_dim=1, hidden_dim=128, output_dim=config["goal_dim"], num=config["num_agents"]).to(device)


    # 初始化 Trainer
    trainer = Trainer(teammateencoder, adhocagentEncoder, returnnet, goaldecoder, lr=config["lr"], alpha=config["alpha"], beta=config["beta"], gama=config["gama"], clip_value=config["clip_value"])
    config["device"] = device
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
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    logger.info("Training Started.")
    # 开始训练
    train_model(logger, trainer, train_loader, val_loader, num_epochs=config["num_epochs"], device=config["device"], test_interval=config["test_interval"],save_interval=config["save_interval"], save_dir=save_dir, model_save_path=config["model_save_path"])
    logger.info("Training completed.")