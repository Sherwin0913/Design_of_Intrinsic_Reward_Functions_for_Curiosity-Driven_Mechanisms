import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
from minigrid.wrappers import ImgObsWrapper
import tqdm

# 设置随机种子保证实验可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 特征编码器
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
        
        # 计算卷积层输出尺寸
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 3, shape[0], shape[1]))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # 调整输入维度顺序 [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2).contiguous()
        conv_out = self.conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)

# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = FeatureEncoder(state_dim)
        self.fc = nn.Linear(512, action_dim)
        
    def forward(self, x):
        features = self.feature(x)
        return self.fc(features)

# ICM模块
class ICMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.feature = FeatureEncoder(state_dim)
        
        # 前向模型
        self.forward_model = nn.Sequential(
            nn.Linear(512 + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512)
        )
        
        # 逆向模型
        self.inverse_model = nn.Sequential(
            nn.Linear(512 * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, state, next_state, action):
        phi_state = self.feature(state)
        phi_next_state = self.feature(next_state)
        
        # 逆向模型
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        # 前向模型
        forward_input = torch.cat([phi_state, F.one_hot(action.long(), num_classes=action_dim).float()], dim=1)
        pred_next_phi = self.forward_model(forward_input)
        
        return pred_next_phi, pred_action, phi_next_state

# DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim, use_icm=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_icm = use_icm
        
        # Q网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        if use_icm:
            self.icm = ICMModel(state_dim, action_dim).to(self.device)
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters())
            
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        self.memory = deque(maxlen=10000)
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            return q_values.argmax().item()
            
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        # 使用numpy.array()先转换为单个数组，再转为tensor，避免慢速警告
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([x[1] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)
        
        # 计算Q值
        current_q = self.q_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        target_q = reward_batch + (1 - done_batch) * 0.99 * next_q
        
        # DQN损失
        q_loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        if self.use_icm:
            # ICM好奇心奖励
            pred_next_phi, pred_action, phi_next_state = self.icm(state_batch, next_state_batch, action_batch)
            forward_loss = F.mse_loss(pred_next_phi, phi_next_state.detach())
            inverse_loss = F.cross_entropy(pred_action, action_batch)
            icm_loss = forward_loss + inverse_loss
            
            # 更新ICM
            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()
            
        # 更新Q网络
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        return q_loss.item()

# 训练函数
def train(env_name, use_icm, episodes):
    env = gym.make(env_name)
    env = ImgObsWrapper(env)  # 将环境包装为图像观察
    state_dim = env.observation_space.shape  # 图像观察的维度
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, use_icm)
    
    returns = []
    losses = []
    
    # 使用tqdm添加进度条
    for episode in tqdm.tqdm(range(episodes), desc="训练进度"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        
        while True:
            action = agent.select_action(state, max(0.01, 0.1 - episode/1000))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.append((state, action, reward, next_state, done))
            loss = agent.train(32)
            
            episode_reward += reward
            if loss:
                episode_loss += loss
            
            if done:
                break
            state = next_state
        
        returns.append(episode_reward)
        losses.append(episode_loss)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Return: {episode_reward:.2f}, Loss: {episode_loss:.4f}")
    
    return returns, losses

if __name__ == "__main__":
    env_name = "MiniGrid-Empty-8x8-v0"  # 使用MiniGrid环境
    episodes = 1000
    
    # DQN与DQN+ICM对比实验
    plt.figure(figsize=(12, 5))
    print("开始训练DQN...")
    dqn_returns, dqn_losses = train(env_name, False, episodes)
    print("开始训练DQN+ICM...")
    dqn_icm_returns, dqn_icm_losses = train(env_name, True, episodes)
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=dqn_returns, label="DQN")
    sns.lineplot(data=dqn_icm_returns, label="DQN+ICM")
    plt.title("DQN vs DQN+ICM Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=dqn_losses, label="DQN")
    sns.lineplot(data=dqn_icm_losses, label="DQN+ICM")
    plt.title("DQN vs DQN+ICM Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig("dqn_comparison.png", dpi=300, bbox_inches='tight')
