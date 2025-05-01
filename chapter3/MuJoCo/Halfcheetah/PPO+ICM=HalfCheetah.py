import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 超参数设置
NUM_EPISODES = 10000  # 确保收敛的episode数量
MAX_STEPS = 1000      # 每个episode的最大步数
GAMMA = 0.99          # 折扣因子
LAMBDA = 0.95         # GAE-Lambda参数
PPO_CLIP = 0.2        # PPO剪裁参数
PPO_EPOCHS = 10       # PPO每次更新的epoch数
BATCH_SIZE = 1024     # 批大小
LEARNING_RATE = 3e-4  # 学习率
ICM_ETA = 0.01        # ICM内在奖励权重
ICM_BETA = 0.2        # ICM前向和逆向模型损失的权重

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # 动作标准差（独立于状态）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # 输出范围 [-1, 1]
        std = torch.exp(self.log_std)    # 确保标准差为正
        return mean, std

# PPO价值网络
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ICM模块
class ICM(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ICM, self).__init__()
        # 特征提取器
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        # 前向模型：预测下一状态特征
        self.forward_model = nn.Sequential(
            nn.Linear(128 + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 逆向模型：预测动作
        self.inverse_model = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

    def forward(self, state, action, next_state):
        state_feat = self.feature(state)
        next_state_feat = self.feature(next_state)
        # 前向模型输入
        forward_input = torch.cat([state_feat, action], dim=-1)
        pred_next_feat = self.forward_model(forward_input)
        # 逆向模型输入
        inverse_input = torch.cat([state_feat, next_state_feat], dim=-1)
        pred_action = self.inverse_model(inverse_input)
        return pred_next_feat, next_state_feat, pred_action

# PPO更新函数
def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs_old, returns, advantages):
    policy_losses, value_losses = [], []
    for _ in range(PPO_EPOCHS):
        for idx in range(0, len(states), BATCH_SIZE):
            batch_idx = slice(idx, min(idx + BATCH_SIZE, len(states)))
            # 将列表转换为 NumPy 数组后再转为张量
            batch_states = torch.FloatTensor(states[batch_idx]).to(device)
            batch_actions = torch.FloatTensor(actions[batch_idx]).to(device)
            batch_log_probs_old = torch.FloatTensor(log_probs_old[batch_idx]).to(device)
            batch_returns = torch.FloatTensor(returns[batch_idx]).to(device)
            batch_advantages = torch.FloatTensor(advantages[batch_idx]).to(device)

            # 策略网络输出
            mean, std = policy_net(batch_states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(batch_actions).sum(dim=-1)

            # PPO损失
            ratio = torch.exp(log_probs - batch_log_probs_old)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值网络损失
            value_pred = value_net(batch_states).squeeze()
            value_loss = (value_pred - batch_returns).pow(2).mean()

            # 更新网络
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)

# ICM更新函数
def icm_update(icm, optimizer_icm, states, actions, next_states):
    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    next_states = torch.FloatTensor(next_states).to(device)

    pred_next_feat, next_feat, pred_action = icm(states, actions, next_states)
    forward_loss = (pred_next_feat - next_feat.detach()).pow(2).mean()
    inverse_loss = (pred_action - actions).pow(2).mean()
    intrinsic_reward = ICM_ETA * forward_loss.item()
    loss = ICM_BETA * forward_loss + (1 - ICM_BETA) * inverse_loss

    optimizer_icm.zero_grad()
    loss.backward()
    optimizer_icm.step()

    return intrinsic_reward

# 计算GAE和回报
def compute_gae(rewards, values, next_value, done):
    advantages = []
    gae = 0
    returns = []
    value = next_value

    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + GAMMA * value * (1 - done) - v
        gae = delta + GAMMA * LAMBDA * gae * (1 - done)
        advantages.insert(0, gae)
        returns.insert(0, gae + v)
        value = v

    return np.array(advantages), np.array(returns)

# 主实验函数
def run_experiment(algo_name, use_icm=False):
    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 初始化网络
    policy_net = PolicyNetwork(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    icm = ICM(obs_dim, act_dim).to(device) if use_icm else None

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    optimizer_icm = optim.Adam(icm.parameters(), lr=LEARNING_RATE) if use_icm else None

    episode_returns, episode_losses = [], []

    # 使用tqdm展示进度
    with tqdm(total=NUM_EPISODES, desc=f"Training {algo_name}") as pbar:
        for episode in range(NUM_EPISODES):
            # 使用 NumPy 数组预分配空间，避免动态追加
            states = np.zeros((MAX_STEPS, obs_dim), dtype=np.float32)
            actions = np.zeros((MAX_STEPS, act_dim), dtype=np.float32)
            rewards = np.zeros(MAX_STEPS, dtype=np.float32)
            log_probs = np.zeros(MAX_STEPS, dtype=np.float32)
            values = np.zeros(MAX_STEPS, dtype=np.float32)

            state, _ = env.reset()  # 提取观察值，忽略info
            episode_reward = 0
            step_count = 0

            for step in range(MAX_STEPS):
                state_tensor = torch.FloatTensor(state).to(device)
                mean, std = policy_net(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = value_net(state_tensor).item()

                next_state, reward, done, truncated, _ = env.step(action.cpu().numpy())
                done = done or truncated
                episode_reward += reward

                # 直接存储到 NumPy 数组中
                states[step] = state
                actions[step] = action.cpu().numpy()
                rewards[step] = reward
                log_probs[step] = log_prob.item()
                values[step] = value
                step_count += 1

                state = next_state
                if done:
                    break

            # 裁剪到实际步数
            states = states[:step_count]
            actions = actions[:step_count]
            rewards = rewards[:step_count]
            log_probs = log_probs[:step_count]
            values = values[:step_count]

            # 计算下一状态价值
            next_state_tensor = torch.FloatTensor(state).to(device)
            next_value = value_net(next_state_tensor).item()

            # 计算GAE和回报
            advantages, returns = compute_gae(rewards, values, next_value, done)

            # 如果使用ICM，计算内在奖励并加到外在奖励上
            if use_icm:
                intrinsic_reward = icm_update(icm, optimizer_icm, states, actions,
                                             np.concatenate([states[1:], [next_state]]))
                rewards = rewards + intrinsic_reward  # NumPy 数组支持广播
                advantages, returns = compute_gae(rewards, values, next_value, done)

            # PPO更新
            policy_loss, value_loss = ppo_update(policy_net, value_net, optimizer_policy, optimizer_value,
                                                 states, actions, log_probs, returns, advantages)

            episode_returns.append(episode_reward)
            episode_losses.append(policy_loss + value_loss)
            pbar.set_postfix({'Return': episode_reward, 'Loss': policy_loss + value_loss})
            pbar.update(1)

    # 保存数据到CSV
    df = pd.DataFrame({'Episode': range(NUM_EPISODES), 'Return': episode_returns, 'Loss': episode_losses})
    df.to_csv(f"{algo_name}_data.csv", index=False)

    return episode_returns, episode_losses

# 可视化函数（顶会风格）
def plot_results(data_ppo, data_ppo_icm, ylabel, filename):
    plt.style.use('seaborn')  # 使用seaborn风格，接近顶会论文
    plt.figure(figsize=(10, 6))

    episodes = range(1, NUM_EPISODES + 1)
    ppo_mean = np.array(data_ppo)
    ppo_icm_mean = np.array(data_ppo_icm)

    plt.plot(episodes, ppo_mean, label='PPO', color='blue', linewidth=2)
    plt.plot(episodes, ppo_icm_mean, label='PPO+ICM', color='orange', linewidth=2)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # 高分辨率保存
    plt.close()

# 主程序
if __name__ == "__main__":
    # 运行实验
    ppo_returns, ppo_losses = run_experiment("PPO", use_icm=False)
    ppo_icm_returns, ppo_icm_losses = run_experiment("PPO_ICM", use_icm=True)

    # 可视化并保存
    plot_results(ppo_returns, ppo_icm_returns, 'Return', 'returns_comparison.png')
    plot_results(ppo_losses, ppo_icm_losses, 'Loss', 'losses_comparison.png')