import torch
import torch.nn as nn

class TeammateEncoder(nn.Module):
    """
    队友编码器
    输入：
        states: 队友的状态，形状为 (num_agents, batch_size, state_dim)
    输出：
        全局编码，形状为 (batch_size, embed_dim)
    """
    def __init__(self, state_dim, embed_dim, num_heads):
        super(TeammateEncoder, self).__init__()
        self.embedding = nn.Linear(state_dim, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.output_layer = nn.Linear(embed_dim, embed_dim)  # 全连接层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化

    def forward(self, states):
        # 输入 states 的形状：(num_agents, batch_size, state_dim)
        
        # embedding
        embedded_states = self.embedding(states)  # (num_agents, batch_size, embed_dim)
        
        # Multihead Attention
        attn_output, _ = self.self_attention(embedded_states, embedded_states, embedded_states)
        
        # transfer to pooling
        attn_output = attn_output.permute(1, 2, 0)  # (batch_size, embed_dim, num_agents)
        
        # average pooling
        pooled_output = self.pooling(attn_output).squeeze(-1)  # (batch_size, embed_dim)
        
        # output
        output = self.output_layer(pooled_output)  # (batch_size, embed_dim)
        
        return output