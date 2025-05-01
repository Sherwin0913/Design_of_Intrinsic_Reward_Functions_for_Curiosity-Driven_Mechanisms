import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适配多进程环境
import matplotlib.pyplot as plt
from collections import deque
import random
from minigrid.wrappers import ImgObsWrapper
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import os

# 设置随机种子保证实验可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 打印显存使用情况（调试用）
def print_memory_usage(device):
    if torch.cuda.is_available() and device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# 特征编码器（共享）
class FeatureEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 3, shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        conv_out = self.conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)

# ICM 模块，使用共享特征编码器的输出
class ICMModel(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_size=256, eta=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.eta = eta

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, phi_state, phi_next_state, action):
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action = self.inverse_model(inverse_input)

        forward_input = torch.cat([phi_state, F.one_hot(action, num_classes=self.action_dim).float()], dim=1)
        pred_next_phi = self.forward_model(forward_input)

        intrinsic_reward = self.eta * F.mse_loss(pred_next_phi, phi_next_state, reduction='none').sum(dim=1)
        return pred_next_phi, pred_action, phi_next_state, intrinsic_reward

# Actor 网络，使用共享特征
class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, features):
        action_probs = F.softmax(self.head(features), dim=-1)
        return torch.distributions.Categorical(action_probs)

# Critic 网络，使用共享特征
class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.head(features).squeeze(-1)

# PPO 智能体
class PPOAgent:
    def __init__(self, state_dim, action_dim, use_icm=False, batch_size=128, learning_rate=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_ratio=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空显存
            print_memory_usage(self.device)  # 打印初始显存使用情况
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_icm = use_icm
        self.batch_size = batch_size  # 减小批次大小
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio

        self.feature = FeatureEncoder(state_dim).to(self.device)
        self.actor = Actor(512, action_dim).to(self.device)
        self.critic = Critic(512).to(self.device)

        if use_icm:
            self.icm = ICMModel(512, action_dim).to(self.device)
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=learning_rate)

        self.actor_optimizer = torch.optim.Adam(list(self.feature.parameters()) + list(self.actor.parameters()), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.feature.parameters()) + list(self.critic.parameters()), lr=learning_rate)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.feature(state)
            dist = self.actor(features)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values[t]
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def train(self, states, actions, rewards, next_states, dones, log_probs):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)

        with torch.no_grad():
            features = self.feature(states)
            next_features = self.feature(next_states)

        if self.use_icm:
            with torch.no_grad():
                _, _, _, intrinsic_rewards = self.icm(features, next_features, actions)
                rewards = rewards + intrinsic_rewards

        with torch.no_grad():
            values = self.critic(features)
            next_values = self.critic(next_features)

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(features, actions, old_log_probs, returns, advantages, next_features, dones)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True, num_workers=0)

        actor_loss_total = 0
        critic_loss_total = 0
        icm_loss_total = 0

        for _ in range(1):
            for batch in dataloader:
                batch_features, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, batch_next_features, batch_dones = batch

                if self.use_icm:
                    pred_next_phi, pred_action, _, _ = self.icm(batch_features, batch_next_features, batch_actions)
                    forward_loss = F.mse_loss(pred_next_phi, batch_next_features.detach())
                    inverse_loss = F.cross_entropy(pred_action, batch_actions)
                    icm_loss = forward_loss + inverse_loss

                    self.icm_optimizer.zero_grad()
                    icm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
                    self.icm_optimizer.step()

                    icm_loss_total += icm_loss.item()

                dist = self.actor(batch_features)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -0.01 * dist.entropy().mean()
                actor_loss += entropy_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.feature.parameters()) + list(self.actor.parameters()), 0.5)
                self.actor_optimizer.step()

                value_pred = self.critic(batch_features)
                critic_loss = F.mse_loss(value_pred, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.feature.parameters()) + list(self.critic.parameters()), 0.5)
                self.critic_optimizer.step()

                actor_loss_total += actor_loss.item()
                critic_loss_total += critic_loss.item()

        return actor_loss_total, critic_loss_total, icm_loss_total if self.use_icm else 0

# 训练函数
def train(env_name, use_icm, episodes):
    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    set_seed(42)

    agent = PPOAgent(state_dim, action_dim, use_icm, batch_size=128)  # 减小批次大小

    returns = []
    actor_losses = []
    critic_losses = []

    num_envs = 1  # 减少并行环境数量，降低显存需求
    envs = [gym.make(env_name) for _ in range(num_envs)]
    envs = [ImgObsWrapper(env) for env in envs]

    states = [env.reset()[0] for env in envs]

    for episode in tqdm(range(0, episodes, num_envs), desc="训练进度"):
        all_states, all_actions, all_rewards, all_next_states, all_dones, all_log_probs = [], [], [], [], [], []
        episode_rewards = [0] * num_envs

        steps_per_env = 128

        for _ in range(steps_per_env):
            actions, log_probs = [], []
            for state in states:
                action, log_prob = agent.select_action(state)
                actions.append(action)
                log_probs.append(log_prob)

            next_states, rewards, dones = [], [], []
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                all_states.append(states[i])
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_states.append(next_state)
                all_dones.append(float(done))
                all_log_probs.append(log_probs[i])

                episode_rewards[i] += reward

                if done:
                    next_state, _ = env.reset()
                    episode_rewards[i] = 0

                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            states = next_states

            if len(all_states) >= 512:  # 减小训练触发阈值
                actor_loss, critic_loss, _ = agent.train(all_states, all_actions, all_rewards, all_next_states,
                                                        all_dones, all_log_probs)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                all_states, all_actions, all_rewards, all_next_states, all_dones, all_log_probs = [], [], [], [], [], []

        if all_states:
            actor_loss, critic_loss, _ = agent.train(all_states, all_actions, all_rewards, all_next_states, all_dones,
                                                    all_log_probs)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if (episode // num_envs) % 10 == 0:
            eval_env = gym.make(env_name)
            eval_env = ImgObsWrapper(eval_env)
            eval_state, _ = eval_env.reset()
            eval_reward = 0
            eval_done = False

            while not eval_done:
                eval_action, _ = agent.select_action(eval_state)
                eval_next_state, eval_r, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                eval_done = eval_terminated or eval_truncated
                eval_reward += eval_r
                eval_state = eval_next_state

            returns.append(eval_reward)

            if ((episode // num_envs) + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Return: {eval_reward:.2f}, Actor Loss: {actor_losses[-1]:.4f}, "
                      f"Critic Loss: {critic_losses[-1]:.4f}")

    for env in envs:
        env.close()

    torch.cuda.empty_cache()  # 训练结束时清空显存
    print_memory_usage(agent.device)  # 打印最终显存使用情况

    return returns, actor_losses, critic_losses

if __name__ == "__main__":
    env_name = "MiniGrid-Empty-8x8-v0"
    episodes = 2500

    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)

    torch.set_num_threads(mp.cpu_count())

    # 清空显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("开始训练 PPO...")
    ppo_returns, ppo_actor_losses, ppo_critic_losses = train(env_name, False, episodes)

    print("开始训练 PPO+ICM...")
    ppo_icm_returns, ppo_icm_actor_losses, ppo_icm_critic_losses = train(env_name, True, episodes)

    # 使用 Matplotlib 绘制曲线
    episodes_axis = np.arange(0, len(ppo_returns)) * 10  # 每 10 个回合记录一次

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episodes_axis, ppo_returns, label="PPO")
    plt.plot(episodes_axis, ppo_icm_returns, label="PPO+ICM")
    plt.title("PPO vs PPO+ICM Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(episodes_axis, ppo_actor_losses[:len(episodes_axis)], label="PPO")  # Ensure same length
    plt.plot(episodes_axis, ppo_icm_actor_losses[:len(episodes_axis)], label="PPO+ICM")  # Ensure same length
    plt.title("PPO vs PPO+ICM Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(episodes_axis, ppo_critic_losses[:len(episodes_axis)], label="PPO")  # Ensure same length
    plt.plot(episodes_axis, ppo_icm_critic_losses[:len(episodes_axis)], label="PPO+ICM")  # Ensure same length
    plt.title("PPO vs PPO+ICM Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.legend()

    plt.tight_layout()

    # 保存图片
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_comparison.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    except Exception as e:
        print(f"保存图片时出错: {e}")

    plt.close()  # 关闭图形，释放内存