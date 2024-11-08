import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Trainer:
    def __init__(self, teammateencoder, adhocagentEncoder, returnnet, goaldecoder, lr=1e-3, alpha=0.1, beta=0.1, clip_value=1.0):
        # 初始化各网络
        self.teammateencoder = teammateencoder
        self.adhocencoder = adhocencoder
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder

        # 单一优化器，联合优化所有模块
        self.optimizer = optim.Adam(
            list(self.teammateencoder.parameters()) + 
            list(self.adhocencoder.parameters()) + 
            list(self.returnnet.parameters()) + 
            list(self.goaldecoder.parameters()), 
            lr=lr
        )
        
        # 设置损失权重
        self.alpha = alpha  # KL损失权重
        self.beta = beta    # r的MSE损失权重
        self.clip_value = clip_value  # 梯度裁剪的最大范数


    def kl_divergence(p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')

    def future_loss(g_theta, p_theta, beta, z, tau, s_t):
        # 第一个 KL 项：D_KL(g_theta(z | τ) || N(0, I))
        # 假设 g_theta 是一个神经网络或函数，输出为 z 的均值和方差
        mu, log_var = g_theta(z, tau)
        std = torch.exp(0.5 * log_var)
        q_z = torch.distributions.Normal(mu, std)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))  # N(0, I)

        kl_term1 = torch.distributions.kl_divergence(q_z, p_z).mean()

        # 第二个 KL 项：D_KL(g_theta(z | τ) || p_theta(z | s_t))，但不计算 g_theta 的梯度
        with torch.no_grad():
            mu_no_grad, log_var_no_grad = g_theta(z, tau)
        std_no_grad = torch.exp(0.5 * log_var_no_grad)
        q_z_no_grad = torch.distributions.Normal(mu_no_grad, std_no_grad)

        mu_st, log_var_st = p_theta(z, s_t)
        std_st = torch.exp(0.5 * log_var_st)
        p_z_st = torch.distributions.Normal(mu_st, std_st)

        kl_term2 = torch.distributions.kl_divergence(q_z_no_grad, p_z_st).mean()

        # 计算最终损失
        loss = beta * kl_term1 + kl_term2
        return loss

    def compute_loss(self, s, o, r_true, g_true):
        # 前向传递
        z = self.teammateencoder(s)
        h = self.adhocencoder(o)
        
        # 计算各损失
        kl_loss = F.kl_div(F.log_softmax(z, dim=-1), F.softmax(h, dim=-1), reduction='batchmean')
        r_pred = self.returnnet(z)
        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        bce_loss_g = nn.BCELoss(g_pred, g_true)

        
        # 总损失，加权组合
        total_loss = self.alpha * kl_loss + self.beta * mse_loss_r + bce_loss_g
        return total_loss, kl_loss, mse_loss_r, bce_loss_g
    
    def train_step(self, s, o, r_true, g_true):
        # 清除梯度
        self.optimizer.zero_grad()

        # 计算损失
        total_loss, kl_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)
        
        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.teammateencoder.parameters()) + 
            list(self.adhocencoder.parameters()) + 
            list(self.returnnet.parameters()) + 
            list(self.goaldecoder.parameters()), 
            max_norm=self.clip_value
        )

        # 优化步骤
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "kl_loss": kl_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }
    
    def evaluate(self, val_loader, device):
        # 切换到评估模式
        self.teamworkencoder.eval()
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        
        total_loss = 0.0
        kl_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for s, o, r_true, g_true in val_loader:
                s, o, r_true, g_true = s.to(device), o.to(device), r_true.to(device), g_true.to(device)
                
                # 计算损失
                loss, kl, mse_r, mse_g = self.compute_loss(s, o, r_true, g_true)
                
                total_loss += loss.item()
                kl_loss += kl.item()
                mse_loss_r += mse_r.item()
                mse_loss_g += mse_g.item()
        
        # 计算验证集上的平均损失
        avg_total_loss = total_loss / num_batches
        avg_kl_loss = kl_loss / num_batches
        avg_mse_loss_r = mse_loss_r / num_batches
        avg_mse_loss_g = mse_loss_g / num_batches
        
        return {
            "avg_total_loss": avg_total_loss,
            "avg_kl_loss": avg_kl_loss,
            "avg_mse_loss_r": avg_mse_loss_r,
            "avg_mse_loss_g": avg_mse_loss_g
        }
