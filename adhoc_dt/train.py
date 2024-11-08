import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import load_config, create_save_directory, setup_logger, save_config
import os

from Networks.ReturnNet import ReturnNet
from Networks.TeamworkEncoder import TeamworkEncoder
from Networks.AdhocAgentEncoder import AdhocAgentEncoder
from Networks.GoalDecoder import GoalDecoder

# 定义训练函数
def train_model(logger, trainer, train_loader, val_loader, num_epochs, device, test_interval, save_interval=5, save_dir, model_save_path="models"):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (s, o, r_true, g_true) in enumerate(pbar):
                s, o, r_true, g_true = s.to(device), o.to(device), r_true.to(device), g_true.to(device)

                loss_dict = trainer.train_step(s, o, r_true, g_true)
                epoch_loss += loss_dict["total_loss"]

                # 更新进度条和打印每个batch的损失
                pbar.set_postfix({
                    "Total Loss": f"{loss_dict['total_loss']:.4f}",
                    "KL Loss": f"{loss_dict['kl_loss']:.4f}",
                    "MSE Loss R": f"{loss_dict['mse_loss_r']:.4f}",
                    "MSE Loss G": f"{loss_dict['mse_loss_g']:.4f}"
                })
                
                # 日志记录
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                                f"Total Loss: {loss_dict['total_loss']:.4f}, KL Loss: {loss_dict['kl_loss']:.4f}, "
                                f"MSE Loss R: {loss_dict['mse_loss_r']:.4f}, MSE Loss G: {loss_dict['mse_loss_g']:.4f}")
        
        # 每个epoch结束后记录平均损失
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Total Loss: {epoch_loss / len(train_loader):.4f}")
        
        val_loss_dict = self.evaluate(val_loader, device)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss_dict['avg_total_loss']:.4f}")
        
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
                    'teamworkencoder': trainer.teamworkencoder.state_dict(),
                    'adhocencoder': trainer.adhocencoder.state_dict(),
                    'returnnet': trainer.returnnet.state_dict(),
                    'goaldecoder': trainer.goaldecoder.state_dict(),
                },
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader)
            }, save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

if __name__ == "__main__":
    env = "PP4a"
    config = load_config(f"./config/{env}_config.yaml")
    save_dir = create_save_directory()
    logger = setup_logger(save_dir)
    
    # 训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 移动模型到设备
    teammateencoder = TeamworkEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"], num_heads=config["TeammateEncoder_num_heads"]).to(device)
    adhocagentEncoder = AdhocAgentEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"]).to(device)
    returnnet = ReturnNet(embed_dim=config["embed_dim"]).to(device)
    goaldecoder = GoalDecoder(embed_dim=config["embed_dim"], goal_dim=config["goal_dim"]).to(device)


    # 初始化 Trainer
    trainer = Trainer(teamworkencoder, adhocagentEncoder, returnnet, goaldecoder, lr=config["lr"], alpha=config["alpha"], beta=config["beta"], clip_value=config["clip_value"])
    config["device"] = device
    # 保存配置到输出文件夹
    save_config(config, save_dir)
    logger.info("Training started.")

    data_path = "data/training_data.pt"  # 假设数据文件已保存为Tensor格式

    # 实例化自定义数据集
    dataset = CustomDataset(data_path)

    # 创建DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 开始训练
    train_model(logger, trainer, train_loader, num_epochs=config["num_epochs"], device=config["device"], test_interval=config["test_interval"],save_interval=config["save_interval"], save_dir=save_dir, model_save_path=config["model_save_path"])
    logger.info("Training completed.")