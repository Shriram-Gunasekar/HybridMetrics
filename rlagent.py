import torch
from torchmetrics import MetricCollection, AverageValueMeter
import gym

class RLAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.metric_collection = MetricCollection({
            'episode_reward': AverageValueMeter(),
            'episode_steps': AverageValueMeter(),
            'success_rate': AverageValueMeter()
        })
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_steps = 0

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1

        if done:
            self.metric_collection['episode_reward'].update(self.episode_reward)
            self.metric_collection['episode_steps'].update(self.episode_steps)
            self.metric_collection['success_rate'].update(int(self.episode_reward > 0))
            self.reset()

        return next_state, reward, done

    def get_metrics(self):
        return self.metric_collection.compute()

# Example usage
agent = RLAgent('CartPole-v1')

for episode in range(100):
    state = agent.state
    done = False

    while not done:
        action = agent.env.action_space.sample()  # Random action for demonstration purposes
        state, reward, done = agent.step(action)

metrics = agent.get_metrics()
print(metrics)
