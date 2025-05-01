import gym  # 或 gymnasium，视环境版本而定
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

# SAC相关超参数
EPISODES = 10000  # 确保收敛的训练轮数
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
MAX_STEPS = 1000

# VIME相关超参数
VIME_BETA = 0.1  # VIME变分信息最大化权重

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SAC Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)
        )
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x[:, :action_dim], x[:, action_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action) * self.max_action
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob


# SAC Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# VIME模块
class VIME(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(VIME, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mean = nn.Linear(128, state_dim)
        self.log_var = nn.Linear(128, state_dim)

    def forward(self, state, action, next_state):
        x = self.encoder(torch.cat([state, action], dim=-1))
        mean = self.mean(x)
        log_var = torch.clamp(self.log_var(x), -20, 2)
        std = torch.exp(0.5 * log_var)
        dist = Normal(mean, std)
        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=-1)
        intrinsic_reward = VIME_BETA * kl_div
        return intrinsic_reward


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, state_dim + action_dim + 1 + state_dim + 1))
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        idx = self.pos % self.capacity
        self.buffer[idx] = np.concatenate([state, action, [reward], next_state, [done]])
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size)
        batch = self.buffer[idx]
        state = torch.FloatTensor(batch[:, :state_dim]).to(device)
        action = torch.FloatTensor(batch[:, state_dim:state_dim + action_dim]).to(device)
        reward = torch.FloatTensor(batch[:, state_dim + action_dim]).to(device)
        next_state = torch.FloatTensor(batch[:, state_dim + action_dim + 1:-1]).to(device)
        done = torch.FloatTensor(batch[:, -1]).to(device)
        return state, action, reward, next_state, done


# SAC训练函数
def train_sac(env, actor, critic1, critic2, target_critic1, target_critic2, log_alpha, buffer, vime=None):
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_opt = optim.Adam([log_alpha], lr=LR_ALPHA)
    alpha = log_alpha.exp().detach()

    returns = []
    losses = []
    results_df = pd.DataFrame(columns=["Episode", "Return", "Loss"])  # 用于保存到CSV

    for episode in tqdm(range(EPISODES), desc="SAC" if vime is None else "SAC+VIME"):
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_return = 0
        episode_loss = 0
        steps = 0

        for _ in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            action, log_prob = actor(state_tensor)
            action = action.cpu().detach().numpy()[0]
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
                done = done or truncated
            else:
                next_state, reward, done, _ = step_result

            if vime:
                intrinsic_reward = vime(
                    state_tensor,
                    torch.FloatTensor(action).unsqueeze(0).to(device),
                    torch.FloatTensor(np.array(next_state)).unsqueeze(0).to(device)
                )
                reward += intrinsic_reward.cpu().detach().numpy()[0]

            buffer.push(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state

            if buffer.size >= BATCH_SIZE:
                state_b, action_b, reward_b, next_state_b, done_b = buffer.sample(BATCH_SIZE)

                # 更新Critic
                with torch.no_grad():
                    next_action, next_log_prob = actor(next_state_b)
                    target_q1 = target_critic1(next_state_b, next_action)
                    target_q2 = target_critic2(next_state_b, next_action)
                    target_q = reward_b.unsqueeze(-1) + GAMMA * (1 - done_b.unsqueeze(-1)) * (
                                torch.min(target_q1, target_q2) - alpha * next_log_prob)

                q1 = critic1(state_b, action_b)
                q2 = critic2(state_b, action_b)
                critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
                critic1_opt.zero_grad()
                critic2_opt.zero_grad()
                critic_loss.backward()
                critic1_opt.step()
                critic2_opt.step()

                # 更新Actor
                action_new, log_prob_new = actor(state_b)
                q1_new = critic1(state_b, action_new)
                q2_new = critic2(state_b, action_new)
                actor_loss = (alpha * log_prob_new - torch.min(q1_new, q2_new)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # 更新Alpha
                alpha_loss = (-log_alpha * (log_prob_new + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                alpha = log_alpha.exp().detach()

                # 软更新目标网络
                for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                episode_loss += critic_loss.item() + actor_loss.item() + alpha_loss.item()
                steps += 1

            if done:
                break

        returns.append(episode_return)
        avg_loss = episode_loss / max(steps, 1)
        losses.append(avg_loss)
        # 将结果追加到DataFrame
        results_df = pd.concat(
            [results_df, pd.DataFrame({"Episode": [episode], "Return": [episode_return], "Loss": [avg_loss]})],
            ignore_index=True)

    # 保存到CSV
    csv_filename = "sac_results.csv" if vime is None else "sac_vime_results.csv"
    results_df.to_csv(csv_filename, index=False)
    return returns, losses


# 可视化函数
def plot_results(data, labels, title, ylabel, filename):
    plt.style.use('seaborn')  # 使用顶刊风格
    plt.figure(figsize=(10, 6))
    for d, label in zip(data, labels):
        plt.plot(d, label=label, linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# 主实验
if __name__ == "__main__":
    # 兼容gym和gymnasium
    try:
        import gymnasium as gym

        env = gym.make('Hopper-v4', render_mode=None)
    except ImportError:
        env = gym.make('Hopper-v4')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    target_entropy = -action_dim

    # SAC基线
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic1 = Critic(state_dim, action_dim).to(device)
    critic2 = Critic(state_dim, action_dim).to(device)
    target_critic1 = Critic(state_dim, action_dim).to(device)
    target_critic2 = Critic(state_dim, action_dim).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)
    sac_returns, sac_losses = train_sac(env, actor, critic1, critic2, target_critic1, target_critic2, log_alpha, buffer)

    # SAC+VIME
    actor_vime = Actor(state_dim, action_dim, max_action).to(device)
    critic1_vime = Critic(state_dim, action_dim).to(device)
    critic2_vime = Critic(state_dim, action_dim).to(device)
    target_critic1_vime = Critic(state_dim, action_dim).to(device)
    target_critic2_vime = Critic(state_dim, action_dim).to(device)
    target_critic1_vime.load_state_dict(critic1_vime.state_dict())
    target_critic2_vime.load_state_dict(critic2_vime.state_dict())
    log_alpha_vime = torch.zeros(1, requires_grad=True, device=device)
    buffer_vime = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)
    vime = VIME(state_dim, action_dim).to(device)
    sac_vime_returns, sac_vime_losses = train_sac(env, actor_vime, critic1_vime, critic2_vime, target_critic1_vime,
                                                  target_critic2_vime, log_alpha_vime, buffer_vime, vime=vime)

    # 可视化
    plot_results([sac_returns, sac_vime_returns], ['SAC', 'SAC+VIME'],
                 'Return Comparison on Hopper-v4', 'Return', 'sacreturns_comparison.png')
    plot_results([sac_losses, sac_vime_losses], ['SAC', 'SAC+VIME'],
                 'Loss Comparison on Hopper-v4', 'Loss', 'saclosses_comparison.png')

    env.close()