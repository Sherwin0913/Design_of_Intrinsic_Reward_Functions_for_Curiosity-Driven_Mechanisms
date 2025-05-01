import os

# 必须添加dll目录，确保MuJoCo正常加载
os.add_dll_directory("C://Users//HP//.mujoco//mjpro150//bin")
os.add_dll_directory("C://Users//HP//.mujoco//mujoco-py-1.50.1.0//mujoco_py")

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------ 离散化动作空间 ------------------------
class Discretizer:
    def __init__(self, action_space, bins_per_dim=5):
        """
        将连续动作空间离散化：
        - 对于每个动作维度，生成 bins_per_dim 个离散取值。
        - 最终离散动作集合为各维度取值的笛卡尔积。
        """
        self.low = action_space.low
        self.high = action_space.high
        self.bins_per_dim = bins_per_dim
        self.dim = action_space.shape[0]
        # 为每个维度生成离散取值
        self.bins = [np.linspace(self.low[i], self.high[i], bins_per_dim) for i in range(self.dim)]
        # 生成所有可能的离散动作
        self.actions = np.array(np.meshgrid(*self.bins)).T.reshape(-1, self.dim)

    @property
    def n(self):
        return len(self.actions)

    def get_action(self, index):
        return self.actions[index]

# ------------------------ 经验回放 ------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

# ------------------------ DQN网络 ------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------ DQN代理 ------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络和目标网络
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.update_steps = 0
        self.target_update_freq = 100  # 每100次更新一次目标网络

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

# ------------------------ ICM模块 ------------------------
# 参考文献: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", 2017.
class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64, lr=1e-3, beta=0.2, eta=0.01):
        """
        beta：平衡前向和逆向损失的权重
        eta：内在奖励的缩放系数
        """
        super(ICMModule, self).__init__()
        self.beta = beta
        self.eta = eta
        self.feature_dim = feature_dim

        # 状态编码器，将原始状态映射到低维特征
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # 逆向模型：根据状态和下一个状态的特征预测动作（分类问题）
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # 前向模型：根据状态特征和动作（one-hot）预测下一个状态特征
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, next_state, action_onehot):
        # 编码状态和下一个状态
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)

        # 逆向模型预测动作
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action_logits = self.inverse_model(inverse_input)

        # 前向模型预测下一个状态特征
        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        pred_phi_next = self.forward_model(forward_input)

        return pred_action_logits, pred_phi_next, phi_next_state

    def compute_loss(self, state, next_state, action, device):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device)

        # 将动作转换成 one-hot 编码
        action_onehot = torch.zeros((action.size(0), self.inverse_model[-1].out_features)).to(device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)

        pred_action_logits, pred_phi_next, phi_next_state = self.forward(state, next_state, action_onehot)

        # 逆向损失：交叉熵损失
        inverse_loss = nn.CrossEntropyLoss()(pred_action_logits, action)
        # 前向损失：均方误差损失
        forward_loss = nn.MSELoss()(pred_phi_next, phi_next_state.detach())

        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        # 内在奖励计算（基于前向预测误差）
        intrinsic_reward = self.eta * 0.5 * torch.sum((pred_phi_next - phi_next_state.detach()) ** 2,
                                                      dim=1).cpu().data.numpy()
        return total_loss, intrinsic_reward

    def update(self, state, next_state, action, device):
        loss, _ = self.compute_loss(state, next_state, action, device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ------------------------ DQN+ICM代理 ------------------------
class DQN_ICM_Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=10000, icm_lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 内部采用 DQNAgent 保存网络与经验回放
        self.dqn_agent = DQNAgent(state_dim, action_dim, lr, gamma, batch_size, buffer_capacity)
        # 初始化ICM模块
        self.icm = ICMModule(state_dim, action_dim, lr=icm_lr).to(self.device)
        self.batch_size = batch_size
        self.gamma = gamma

    def select_action(self, state, epsilon):
        return self.dqn_agent.select_action(state, epsilon)

    def update(self):
        # 确保经验数量足够
        if len(self.dqn_agent.replay_buffer) < self.batch_size:
            return 0, 0
        batch = self.dqn_agent.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 先更新ICM模块，并计算内在奖励
        icm_loss = self.icm.update(states, next_states, actions, self.device)
        with torch.no_grad():
            _, intrinsic_rewards = self.icm.compute_loss(states, next_states, actions, self.device)
        # 组合奖励：外部奖励+内在奖励
        combined_rewards = np.array(rewards) + intrinsic_rewards

        # 接下来使用组合奖励更新 DQN 部分
        states_tensor = torch.FloatTensor(np.array(states)).to(self.dqn_agent.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.dqn_agent.device)
        rewards_tensor = torch.FloatTensor(combined_rewards).unsqueeze(1).to(self.dqn_agent.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.dqn_agent.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.dqn_agent.device)

        q_values = self.dqn_agent.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.dqn_agent.target_net(next_states_tensor).max(1, keepdim=True)[0]
        expected_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
        dqn_loss = nn.MSELoss()(q_values, expected_q_values)

        self.dqn_agent.optimizer.zero_grad()
        dqn_loss.backward()
        self.dqn_agent.optimizer.step()

        self.dqn_agent.update_steps += 1
        if self.dqn_agent.update_steps % self.dqn_agent.target_update_freq == 0:
            self.dqn_agent.target_net.load_state_dict(self.dqn_agent.policy_net.state_dict())
        return dqn_loss.item(), icm_loss

# ------------------------ 训练函数 ------------------------
def run_dqn_training(agent, env, discretizer, num_episodes=300, max_steps=200, initial_epsilon=1.0, min_epsilon=0.05,
                     epsilon_decay=0.995):
    """
    纯DQN训练过程，返回每个episode的总reward和平均loss
    """
    returns = []
    losses = []
    for episode in tqdm(range(num_episodes), desc="DQN Episodes"):
        state, _ = env.reset()
        episode_return = 0
        episode_losses = []
        # 衰减式epsilon
        epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))
        for step in range(max_steps):
            action_idx = agent.select_action(state, epsilon)
            # 将离散动作索引转换为连续动作
            continuous_action = discretizer.get_action(action_idx)
            next_state, reward, done, truncated, info = env.step(continuous_action)
            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            loss = agent.update()
            if loss:
                episode_losses.append(loss)
            state = next_state
            episode_return += reward
            if done:
                break
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        returns.append(episode_return)
        losses.append(avg_loss)
    return returns, losses


def run_dqn_icm_training(agent, env, discretizer, num_episodes=300, max_steps=200, initial_epsilon=1.0, min_epsilon=0.05,
                         epsilon_decay=0.995):
    """
    DQN+ICM训练过程，返回每个episode的总reward和平均loss
    """
    returns = []
    losses = []
    for episode in tqdm(range(num_episodes), desc="DQN+ICM Episodes"):
        state, _ = env.reset()
        episode_return = 0
        episode_losses = []
        epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))
        for step in range(max_steps):
            action_idx = agent.select_action(state, epsilon)
            continuous_action = discretizer.get_action(action_idx)
            # 先将transition存入经验回放（暂时reward设为0，后续覆盖）
            agent.dqn_agent.replay_buffer.push(state, action_idx, 0, None, None)
            next_state, reward, done, truncated, info = env.step(continuous_action)
            # 更新刚刚存入的transition
            agent.dqn_agent.replay_buffer.buffer[-1] = (state, action_idx, reward, next_state, done)
            dqn_loss, icm_loss = agent.update()
            total_loss = dqn_loss + icm_loss
            if total_loss:
                episode_losses.append(total_loss)
            state = next_state
            episode_return += reward
            if done:
                break
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        returns.append(episode_return)
        losses.append(avg_loss)
    return returns, losses

# ------------------------ 结果保存与可视化 ------------------------
def plot_results(returns, losses, title_prefix):
    plt.style.use('seaborn-paper')

    # 绘制Return曲线
    plt.figure(figsize=(6, 4))
    plt.plot(returns, label="Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{title_prefix} Return over Episodes")
    plt.grid(True)
    plt.legend()
    png_name = f"{title_prefix.lower().replace(' ', '_')}_return.png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.close()

    # 绘制Loss曲线
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Loss", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss over Episodes")
    plt.grid(True)
    plt.legend()
    png_name = f"{title_prefix.lower().replace(' ', '_')}_loss.png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_csv(returns, losses, filename):
    df = pd.DataFrame({
        "episode": np.arange(len(returns)),
        "return": returns,
        "loss": losses
    })
    df.to_csv(filename, index=False)

# ------------------------ 主函数 ------------------------
def main():
    # 创建MuJoCo环境，注意render_mode设置为'human'
    env = gym.make("Reacher-v4", render_mode='human')

    # 离散化连续动作空间
    if isinstance(env.action_space, gym.spaces.Box):
        discretizer = Discretizer(env.action_space, bins_per_dim=5)
        action_dim = discretizer.n
    else:
        discretizer = None
        action_dim = env.action_space.n

    # 根据环境信息获取状态维度
    state_dim = env.observation_space.shape[0]

    num_episodes = 10000  # 可根据实际情况调大，确保return收敛
    max_steps = 200  # 每个episode最大步数

    # ------------------------ 实验1：纯DQN ------------------------
    dqn_agent = DQNAgent(state_dim, action_dim)
    returns_dqn, losses_dqn = run_dqn_training(dqn_agent, env, discretizer, num_episodes, max_steps)
    save_results_csv(returns_dqn, losses_dqn, "dqn_results.csv")
    plot_results(returns_dqn, losses_dqn, "DQN")

    # ------------------------ 实验2：DQN+ICM ------------------------
    dqn_icm_agent = DQN_ICM_Agent(state_dim, action_dim)
    returns_dqn_icm, losses_dqn_icm = run_dqn_icm_training(dqn_icm_agent, env, discretizer, num_episodes, max_steps)
    save_results_csv(returns_dqn_icm, losses_dqn_icm, "dqn_icm_results.csv")
    plot_results(returns_dqn_icm, losses_dqn_icm, "DQN ICM")

    env.close()

if __name__ == '__main__':
    main()
