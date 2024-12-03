from Networks.dt_models.decision_transformer import DecisionTransformer
import torch.nn.functional as F
import torch
class DtAgent:
    def __init__(self, dt_model, env_type='PP4a'):
        self.dt_model = dt_model
        self.env_type = env_type

    def take_action(self, o_list, a_list, R_list, t_list, act_dim):

        self.dt_model.eval()

        # 将列表转换为tensor list
        o_list = [torch.tensor(o).to("cuda").clone().detach() for o in o_list]
        a_list = [torch.tensor(a).to("cuda").clone().detach() for a in a_list]
        R_list = [torch.tensor(R).to("cuda").clone().detach() for R in R_list]
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
        '''
        if self.env_type == "LBF":
            o = o[..., [-3, -2]]
        '''
        R_list = [r.unsqueeze(0) if r.dim() == 0 else r for r in R_list]
        R = torch.cat(R_list, dim=0).unsqueeze(0)
        o = o.unsqueeze(0)
        a = a.unsqueeze(0)
        t = t.unsqueeze(0)
        # print(new_g.shape)
        # print(a.shape)
        # print(o.shape)
        # print(t)
        a_onehot = F.one_hot(a.to(torch.int64), num_classes=act_dim)
        ac_pred = self.dt_model.get_action(o, a_onehot, R, t)
        ac_pred = torch.argmax(ac_pred)
        # print(ac_pred)
        return [ac_pred.detach().cpu().numpy()]