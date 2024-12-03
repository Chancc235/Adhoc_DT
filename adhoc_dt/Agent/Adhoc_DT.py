from Networks.dt_models.decision_transformer import DecisionTransformer
import torch.nn.functional as F
import torch
class Adhoc_DT:
    def __init__(self, dt_model, state_encoder, return_net, goal_decoder, env_type='PP4a'):
        self.state_encoder = state_encoder
        self.dt_model = dt_model
        self.return_net = return_net
        self.goal_decoder = goal_decoder
        self.env_type = env_type


    def take_action(self, o_list, a_list, g_list, t_list, act_dim):

        self.state_encoder.eval()
        self.dt_model.eval()
        self.return_net.eval()
        self.goal_decoder.eval()

        # 将列表转换为tensor list
        o_list = [torch.tensor(o).to("cuda").clone().detach() for o in o_list]
        a_list = [torch.tensor(a).to("cuda").clone().detach() for a in a_list]
        g_list = [torch.tensor(g).to("cuda").clone().detach() for g in g_list]
        t_list = [torch.tensor(t).to("cuda").clone().detach() for t in t_list]
        # 当前时间步的状态
        obs = o_list[0].view(1, o_list[0].shape[0])
        # 转换为可用的o ,a, s 序列
        o = torch.cat(o_list, dim=0)
        if len(a_list) != 0:
            a = torch.tensor(a_list).to(device="cuda")
        else:
            a = torch.zeros(len(o_list), 1).to(device="cuda")
        o = o.to(device="cuda")
        o = o.view(len(o_list), o_list[0].shape[0])
        t = torch.tensor(t_list).to(device="cuda")

        # 计算状态embeding
        if self.env_type == "LBF":
            obs = obs[..., [-3, -2]]
        z_mu, z_log_var = self.state_encoder(obs)
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        z_std = torch.exp(0.5 * z_log_var)
        z = torch.distributions.Normal(z_mu, z_std).sample()
        # 计算奖励
        r = self.return_net(z)
        # 预测子目标
        if self.env_type == "PP4a":
            pred_g = (self.goal_decoder(z, r) > 0.5).float().permute(1, 0, 2)
        elif self.env_type == "LBF":
            pred_g = self.goal_decoder(z, r).permute(1, 0, 2)
        # 拼接新的子目标
        g_list.append(pred_g)
        new_g = torch.cat(g_list, dim=0)
        o = o.unsqueeze(0)
        a = a.unsqueeze(0)
        t = t.unsqueeze(0)
        # print(new_g.shape)
        # print(a.shape)
        # print(o.shape)
        # print(t)
        a_onehot = F.one_hot(a.to(torch.int64), num_classes=act_dim)
        ac_pred = self.dt_model.get_action(o, a_onehot, new_g, t)
        ac_pred = torch.argmax(ac_pred)
        # print(ac_pred)
        return [ac_pred.detach().cpu().numpy()], pred_g