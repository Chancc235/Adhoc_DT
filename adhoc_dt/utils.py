import logging
import os
from datetime import datetime
import yaml
import torch
import numpy as np
import time

# 定义用于保存模型和日志的文件夹
def create_save_directory(base_dir="training_output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# 设置日志记录
def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "training_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

# 保存配置到输出文件夹
def save_config(config, save_dir):
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def print_elapsed_time(start_time, logger):
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Time passed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


# 从episode_data加载数据
def load_data(data_path, num_agents = 4):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    start_time = time.time()
    logger.info("Start load data")
    episodes_data = torch.load(data_path)
    logger.info("loaded data")
    obs, actions, states, next_states, rewards = [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)], [[] for _ in range(num_agents)]
    length = len(episodes_data)
    cnt = 0
    for episode in episodes_data:
        cnt += 1
        if cnt % 100 == 0:
            logger.info(f"load data {cnt} / {length}")
        if cnt % 1000 == 0:
            print_elapsed_time(start_time, logger)
        # 找到每个 episode 的有效长度（未达到 done 的步数）
        episode_length = np.where(episode["done"] == 1)[0]
        episode_length = episode_length[0] if len(episode_length) > 0 else episode["done"].shape[0]
        
        # 批量提取各 agent 的数据
        for i in range(num_agents):
            # 使用切片批量添加数据，避免逐步循环
            states[i].extend(episode["state"][:episode_length])
            next_states[i].extend(episode["next_state"][:episode_length])
            obs[i].extend(episode["state"][:episode_length, i])
            actions[i].extend(episode["action"][:episode_length, i])
            rewards[i].extend(episode["reward"][:episode_length])

    # 将所有数据转换为 NumPy 数组
    states = [np.array(s) for agent_states in states for s in agent_states]
    next_states = [np.array(n) for agent_next_states in next_states for n in agent_next_states]
    obs = [np.array(o) for agent_obs in obs for o in agent_obs ]
    actions = [np.array(a) for agent_actions in actions for a in agent_actions]
    rewards = [np.array(r) for agent_rewards in rewards for r in agent_rewards]
    logger.info(f"load data done")

    return states, next_states, obs, actions, rewards

