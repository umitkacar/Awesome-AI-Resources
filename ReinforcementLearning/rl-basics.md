# Reinforcement Learning Basics

A comprehensive introduction to Reinforcement Learning concepts, algorithms, and implementations.

**Last Updated:** 2025-06-19

## Table of Contents
- [What is Reinforcement Learning?](#what-is-reinforcement-learning)
- [Key Concepts](#key-concepts)
- [Classic Algorithms](#classic-algorithms)
- [Getting Started](#getting-started)
- [Popular Environments](#popular-environments)
- [Learning Resources](#learning-resources)
- [Implementation Examples](#implementation-examples)

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a paradigm where agents learn to make decisions by interacting with an environment to maximize cumulative reward.

### Core Components
1. **Agent**: The learner and decision maker
2. **Environment**: What the agent interacts with
3. **State**: Current situation
4. **Action**: What the agent can do
5. **Reward**: Feedback signal
6. **Policy**: Agent's behavior strategy

### RL vs Other ML Paradigms
- **Supervised Learning**: Learning from labeled examples
- **Unsupervised Learning**: Finding patterns in data
- **Reinforcement Learning**: Learning from interaction

## Key Concepts

### Markov Decision Process (MDP)
**Mathematical Framework:**
- States (S)
- Actions (A)
- Transition probabilities P(s'|s,a)
- Rewards R(s,a,s')
- Discount factor γ

### Value Functions
**State Value Function V(s):**
- Expected return from state s
- Following policy π

**Action Value Function Q(s,a):**
- Expected return from state s
- Taking action a, then following π

### Exploration vs Exploitation
**Balancing Strategies:**
- ε-greedy
- Upper Confidence Bound (UCB)
- Thompson Sampling
- Boltzmann exploration

## Classic Algorithms

### Dynamic Programming
**[Value Iteration](https://github.com/dennybritz/reinforcement-learning/tree/master/DP)** - When model is known
- 🟢 Beginner friendly
- Guaranteed convergence
- Computationally expensive
- Full environment model needed

**Policy Iteration:**
- Policy evaluation + improvement
- Faster convergence
- More stable

### Monte Carlo Methods
**[MC Prediction & Control](https://github.com/openai/gym)** - Learn from episodes
- Model-free
- Episodic tasks only
- High variance
- Simple to understand

### Temporal Difference Learning
**Q-Learning:**
```python
# Q-Learning update
Q[s,a] = Q[s,a] + α * (r + γ * max(Q[s']) - Q[s,a])
```

**SARSA:**
```python
# SARSA update
Q[s,a] = Q[s,a] + α * (r + γ * Q[s',a'] - Q[s,a])
```

## Getting Started

### Python Libraries
**[OpenAI Gym](https://gym.openai.com/)** - RL environments
- 🆓 Open source
- Standard interface
- Diverse environments
- Easy visualization

**[Stable Baselines3](https://stable-baselines3.readthedocs.io/)** - RL algorithms
- 🟢 Beginner friendly
- PyTorch based
- Well documented
- Production ready

**[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)** - Scalable RL
- 🔴 Advanced
- Distributed training
- Multi-agent support
- Production systems

### Quick Start Example
```python
import gym
import numpy as np

# Create environment
env = gym.make('CartPole-v1')

# Random agent
for episode in range(10):
    observation = env.reset()
    total_reward = 0
    
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode {episode} finished after {t+1} timesteps")
            print(f"Total reward: {total_reward}")
            break

env.close()
```

## Popular Environments

### Classic Control
**CartPole:**
- Balance pole on cart
- Discrete actions
- Great for beginners
- Quick training

**MountainCar:**
- Reach goal with momentum
- Continuous/discrete
- Exploration challenge
- Sparse rewards

**Pendulum:**
- Swing up pendulum
- Continuous control
- Torque limits
- Classic benchmark

### Atari Games
**[Atari 2600](https://github.com/openai/gym)** - Pixel-based learning
- 57 games
- Visual input
- Discrete actions
- DQN benchmark

### Robotics
**[MuJoCo](https://mujoco.org/)** - Physics simulation
- 💰 Now free!
- Continuous control
- Complex dynamics
- Research standard

**[PyBullet](https://pybullet.org/)** - Open source alternative
- 🆓 Free
- Robot simulation
- Good physics
- Active community

### Custom Environments
**[gym-examples](https://github.com/Farama-Foundation/gym-examples)** - Template
- Environment creation
- Registration process
- Best practices
- Documentation

## Learning Resources

### Online Courses
**[Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning)** - University of Alberta
- 🟡 Intermediate
- Sutton & Barto based
- Comprehensive theory
- Programming assignments

**[Deep RL Course](https://huggingface.co/deep-rl-course/unit0/introduction)** - Hugging Face
- 🆓 Free
- Hands-on approach
- Modern algorithms
- Community support

**[Spinning Up in Deep RL](https://spinningup.openai.com/)** - OpenAI
- Research focused
- Clear explanations
- Code implementations
- Educational resource

### Books
**[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)** - Sutton & Barto
- 🆓 Free online
- RL bible
- Comprehensive coverage
- Mathematical rigor

**[Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning)** - Miguel Morales
- 💰 Paid
- Intuitive approach
- Visual explanations
- Practical focus

### Video Lectures
**[David Silver's RL Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)** - DeepMind
- 🆓 Free
- UCL lectures
- DeepMind researcher
- Comprehensive

**[CS285 Berkeley](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)** - Deep RL
- Advanced topics
- Latest research
- Sergey Levine
- Policy gradients

## Implementation Examples

### Tabular Q-Learning
```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
```

### Simple Policy Gradient
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

def reinforce_update(policy, optimizer, rewards, log_probs):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
    
    loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        loss.append(-log_prob * reward)
    
    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()
```

## Common Challenges

### Sample Efficiency
**Problems:**
- RL needs lots of data
- Real-world interaction expensive
- Simulation gaps

**Solutions:**
- Model-based RL
- Transfer learning
- Curriculum learning
- Data augmentation

### Stability
**Issues:**
- Non-stationary targets
- Function approximation
- Deadly triad

**Techniques:**
- Target networks
- Experience replay
- Gradient clipping
- Proper initialization

### Exploration
**Challenges:**
- Local optima
- Sparse rewards
- Large state spaces

**Methods:**
- Intrinsic motivation
- Count-based exploration
- Prediction error
- Information gain

## Next Steps

### Advanced Topics
1. **[Deep RL](./deep-rl.md)** - Neural networks in RL
2. **[Multi-Agent RL](./multi-agent-rl.md)** - Multiple agents
3. **Model-Based RL** - Learning environment models
4. **Inverse RL** - Learning from demonstrations
5. **Meta-RL** - Learning to learn

### Research Directions
- Offline RL
- Safe RL
- Hierarchical RL
- Continual RL
- Real-world applications

### Community Resources
**Forums & Discussion:**
- [r/reinforcementlearning](https://reddit.com/r/reinforcementlearning)
- [RL Discord](https://discord.gg/reinforcement-learning)
- [Stack Overflow RL tag](https://stackoverflow.com/questions/tagged/reinforcement-learning)

**Competitions:**
- [NeurIPS Competitions](https://neurips.cc/Conferences/2024/CompetitionTrack)
- [AIcrowd RL Challenges](https://www.aicrowd.com/)
- [Kaggle RL Competitions](https://www.kaggle.com/competitions)