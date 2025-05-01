# 用ppo+rnd算法 训练agent玩atari游戏
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 超参数
NUM_ENVS = 16  # 并行环境数量
NUM_STEPS = 256  # 每次采集的步数
TOTAL_TIMESTEPS = 100000000  # 总时间步数
LEARNING_RATE = 3e-4  # 学习率
GAMMA = 0.99  # 折扣因子（外在奖励）
GAMMA_INT = 0.99  # 折扣因子（内在奖励）
BETA = 1.0  # 内在奖励缩放因子
GAE_LAMBDA = 0.95  # GAE 参数
CLIP_EPS = 0.2  # PPO 剪切范围
ENT_COEF = 0.01  # 熵系数
VF_COEF = 0.5  # 值函数系数

# 创建 Atari 环境
def make_env():
    def _thunk():
        env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
        env = gym.wrappers.AtariPreprocessing(
            env, frame_skip=4, screen_size=64, grayscale_obs=True, scale_obs=True
        )
        env = gym.wrappers.FrameStack(env, 4)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # 添加 episode 统计
        return env
    return _thunk

# 策略网络（调整为适配 64x64 输入）
class Policy(nn.Module):
    def __init__(self, action_dim, use_rnd=False):
        super(Policy, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(1024, 512), nn.ReLU()
        )
        self.policy_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)
        self.use_rnd = use_rnd
        if use_rnd:
            self.value_head_int = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        features = self.cnn(x)
        logits = self.policy_head(features)
        value_ext = self.value_head(features)
        if self.use_rnd:
            value_int = self.value_head_int(features)
            return logits, value_ext, value_int
        return logits, value_ext

# RND 模块（同样调整为适配 64x64 输入）
class RND(nn.Module):
    def __init__(self):
        super(RND, self).__init__()
        self.target = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(1024, 512)
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 512)
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.float() / 255.0
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature

# 观察标准化器
class RunningStat:
    def __init__(self, shape, device):
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_std = torch.std(x, dim=0)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        m_a = self.std * self.count
        m_b = batch_std * batch_count
        M2 = m_a * m_b + (delta ** 2) * self.count * batch_count / (self.count + batch_count)
        self.std = torch.sqrt(M2 / (self.count + batch_count) + 1e-6)
        self.count += batch_count

    def normalize(self, x):
        return torch.clamp((x - self.mean) / self.std, -5, 5)

# PPO 代理
class PPO:
    def __init__(self, envs, with_rnd=False):
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.with_rnd = with_rnd
        self.policy = Policy(envs.single_action_space.n, with_rnd).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        if with_rnd:
            self.rnd = RND().to(self.device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=LEARNING_RATE)
            self.obs_normalizer = RunningStat((4, 64, 64), self.device)

    def collect_rollouts(self):
        obs, _ = self.envs.reset()
        rollouts = {'obs': [], 'actions': [], 'rewards': [], 'values_ext': [], 'log_probs': [],
                    'dones': [], 'next_obs': None}
        if self.with_rnd:
            rollouts['values_int'] = []
            rollouts['int_rewards'] = []
        for _ in range(NUM_STEPS):
            obs_array = np.stack(obs) if isinstance(obs, list) else obs
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if self.with_rnd:
                    logits, value_ext, value_int = self.policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    predict, target = self.rnd(obs_tensor)
                    int_reward = torch.mean((predict - target) ** 2, dim=1)
                else:
                    logits, value_ext = self.policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            done = terminated | truncated
            rollouts['obs'].append(obs_tensor)
            rollouts['actions'].append(action)
            rollouts['rewards'].append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            rollouts['values_ext'].append(value_ext.squeeze(-1))
            rollouts['log_probs'].append(log_prob)
            rollouts['dones'].append(torch.tensor(done, dtype=torch.float32, device=self.device))
            if self.with_rnd:
                rollouts['values_int'].append(value_int.squeeze(-1))
                rollouts['int_rewards'].append(int_reward)
                self.obs_normalizer.update(obs_tensor)
            obs = next_obs
            for i, d in enumerate(done):
                if d:
                    if isinstance(info, list) and i < len(info):
                        print(f"Env {i} done, info: {info[i]}")
                        if isinstance(info[i], dict) and 'episode' in info[i]:
                            rollouts.setdefault('episodes', []).append({
                                'return': info[i]['episode']['r'],
                                'timestep': len(rollouts['rewards']) * NUM_ENVS + i
                            })
                    elif isinstance(info, dict):
                        env_key = str(i)
                        if env_key in info:
                            print(f"Env {i} done, info: {info[env_key]}")
                            if isinstance(info[env_key], dict) and 'episode' in info[env_key]:
                                rollouts.setdefault('episodes', []).append({
                                    'return': info[env_key]['episode']['r'],
                                    'timestep': len(rollouts['rewards']) * NUM_ENVS + i
                                })
        next_obs_array = np.stack(obs) if isinstance(obs, list) else obs
        rollouts['next_obs'] = torch.tensor(next_obs_array, dtype=torch.float32, device=self.device)
        return rollouts

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return torch.stack(advantages)

    def update(self, rollouts):
        with torch.no_grad():
            if self.with_rnd:
                _, next_value_ext, next_value_int = self.policy(rollouts['next_obs'])
                next_value_ext = next_value_ext.squeeze(-1)
                next_value_int = next_value_int.squeeze(-1)
                total_rewards = [r + BETA * ir for r, ir in zip(rollouts['rewards'], rollouts['int_rewards'])]
                advantages = self.compute_gae(total_rewards, rollouts['values_ext'], next_value_ext, rollouts['dones'])
                returns_ext = self.compute_gae(rollouts['rewards'], rollouts['values_ext'], next_value_ext,
                                               rollouts['dones']) + torch.stack(rollouts['values_ext'])
                returns_int = self.compute_gae(rollouts['int_rewards'], rollouts['values_int'], next_value_int,
                                               rollouts['dones']) + torch.stack(rollouts['values_int'])
            else:
                _, next_value_ext = self.policy(rollouts['next_obs'])
                next_value_ext = next_value_ext.squeeze(-1)
                advantages = self.compute_gae(rollouts['rewards'], rollouts['values_ext'], next_value_ext,
                                              rollouts['dones'])
                returns_ext = advantages + torch.stack(rollouts['values_ext'])

            advantages = advantages.view(-1)
            if self.with_rnd:
                returns_ext = returns_ext.view(-1)
                returns_int = returns_int.view(-1)
            else:
                returns_ext = returns_ext.view(-1)

        obs = torch.cat(rollouts['obs'])
        actions = torch.cat(rollouts['actions'])
        old_log_probs = torch.cat(rollouts['log_probs'])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss, value_loss_ext, value_loss_int = 0, 0, 0
        for _ in range(10):
            if self.with_rnd:
                logits, value_ext, value_int = self.policy(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * entropy
                value_loss_ext = F.mse_loss(value_ext.squeeze(-1), returns_ext)
                value_loss_int = F.mse_loss(value_int.squeeze(-1), returns_int)
                loss = policy_loss + VF_COEF * (value_loss_ext + value_loss_int)
            else:
                logits, value_ext = self.policy(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * entropy
                value_loss_ext = F.mse_loss(value_ext.squeeze(-1), returns_ext)
                loss = policy_loss + VF_COEF * value_loss_ext

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.with_rnd:
                predict, target = self.rnd(obs)
                rnd_loss = torch.mean((predict - target) ** 2)
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

        return {'policy_loss': policy_loss.item(), 'value_loss_ext': value_loss_ext.item(),
                'value_loss_int': value_loss_int.item() if self.with_rnd else None}

# 训练函数
def train(with_rnd, total_timesteps, name):
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    agent = PPO(envs, with_rnd)
    episode_data = []
    update_data = []
    total_steps = 0

    log_dir = f"runs/{name}"
    writer = SummaryWriter(log_dir=log_dir)
    episode_return_tag = 'Return/Episode/RND' if with_rnd else 'Return/Episode/NoRND'
    rollout_return_tag = 'Return/Rollout/RND' if with_rnd else 'Return/Rollout/NoRND'

    with tqdm(total=total_timesteps, desc=f"Training {'with RND' if with_rnd else 'without RND'}") as pbar:
        update_step = 0
        while total_steps < total_timesteps:
            rollouts = agent.collect_rollouts()
            steps = NUM_STEPS * NUM_ENVS
            total_steps += steps
            pbar.update(steps)
            losses = agent.update(rollouts)
            update_data.append({'total_timesteps': total_steps, **losses})
            update_step += 1

            writer.add_scalar('Loss/Policy', losses['policy_loss'], update_step)
            writer.add_scalar('Loss/Value_Ext', losses['value_loss_ext'], update_step)
            if with_rnd and losses['value_loss_int'] is not None:
                writer.add_scalar('Loss/Value_Int', losses['value_loss_int'], update_step)

            rollout_reward = torch.stack(rollouts['rewards']).sum().item()
            writer.add_scalar(rollout_return_tag, rollout_reward, update_step)

            if 'episodes' in rollouts:
                for ep in rollouts['episodes']:
                    episode_data.append({'total_timesteps': ep['timestep'], 'return': ep['return']})
                    writer.add_scalar(episode_return_tag, ep['return'], update_step)

    envs.close()
    writer.close()
    return episode_data, update_data

# 绘图函数
def plot_comparison(data_rnd, data_no_rnd, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    sns.lineplot(x=[d['total_timesteps'] for d in data_rnd],
                 y=[d['return'] if ylabel == 'Return' else d['policy_loss'] for d in data_rnd], label='With RND',
                 ci=None)
    sns.lineplot(x=[d['total_timesteps'] for d in data_no_rnd],
                 y=[d['return'] if ylabel == 'Return' else d['policy_loss'] for d in data_no_rnd], label='Without RND',
                 ci=None)
    plt.xlabel('Total Timesteps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 主程序
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires a GPU.")

    episode_data_rnd, update_data_rnd = train(with_rnd=True, total_timesteps=TOTAL_TIMESTEPS, name="rnd")
    episode_data_no_rnd, update_data_no_rnd = train(with_rnd=False, total_timesteps=TOTAL_TIMESTEPS, name="no_rnd")

    pd.DataFrame(episode_data_rnd).to_csv('episode_returns_rnd.csv', index=False)
    pd.DataFrame(episode_data_no_rnd).to_csv('episode_returns_no_rnd.csv', index=False)
    pd.DataFrame(update_data_rnd).to_csv('update_losses_rnd.csv', index=False)
    pd.DataFrame(update_data_no_rnd).to_csv('update_losses_no_rnd.csv', index=False)

    plot_comparison(episode_data_rnd, episode_data_no_rnd, 'Return', 'Episode Returns Comparison', 'returns.png')
    plot_comparison(update_data_rnd, update_data_no_rnd, 'Policy Loss', 'Policy Loss Comparison', 'losses.png')

    print("实验完成，结果已保存到当前目录下。")
    print("请使用以下命令启动 TensorBoard 查看训练过程：")
    print("tensorboard --logdir runs")