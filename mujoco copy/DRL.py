import torch
from PPOAgent import PPOAgent
from torch.optim import Adam
import gymnasium as gym
from customEnv import MyEnv
from TrajData import TrajData
# from ReplayBuffer import ReplayBuffer
from tqdm import tqdm
import numpy as np

class DRL:
    def __init__(self):

        self.n_envs = 16 
        self.n_steps = 1000 # about 1 sim sec
        self.n_obs = 35+34+160+1 # qpos+qvel+cinert+1(phase)
        self.n_actions = 28 # 28 actuators, each action modeled as a gaussian

        self.envs = gym.vector.SyncVectorEnv([lambda: MyEnv("test.xml") for _ in range(self.n_envs)])

        # self.replay_buffer = ReplayBuffer()
        self.traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions, 0,0) # placeholder for ind, maybe no longer in use
  
        self.agent = PPOAgent(self.n_obs, n_actions=self.n_actions, a_lambda=.95, gamma = .99)  
        # self.optimizer = Adam(self.agent.parameters(), lr=1e-3)
        self.actor_optimizer = Adam(self.agent.policy.parameters(), lr=1e-4)
        self.critic_optimizer = Adam(self.agent.value.parameters(),lr=1e-4)
        # self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i):

        obs, _ = self.envs.reset() # obs, reset_info
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            # print(actions.shape)
            log_probs = probs.log_prob(actions).sum(-1)

            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            # self.replay_buffer.store(obs,actions,rewards,next_obs,done,traj_ind,t)
            obs = torch.Tensor(next_obs)
                

        self.traj_data.calc_returns()
        # self.traj_datas.append(traj_data)
        # print("avg reward: ", self.traj_data.rewards.mean())
        self.avg_reward = self.traj_data.rewards.mean()

        # self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        # self.writer.flush()

    def get_avg_loss(self):
        return self.avg_policy_loss,self.avg_value_loss
    
    def get_avg_reward(self):
        return self.avg_reward
    
    def get_avg_sim_steps(self):
        return self.avg_sim_steps


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 10 if self.agent.name == 'PPO' else 1
        epoch_policy_loss = []
        epoch_value_loss = []

        for _ in range(epochs):

            policy_loss,value_loss = self.agent.get_loss(self.traj_data)
            # print(f"policy loss: {policy_loss.item()}, value loss: {value_loss.item()}")
            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph = True)
            policy_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        self.traj_data.detach()
        self.avg_policy_loss = np.mean(epoch_policy_loss)
        self.avg_value_loss = np.mean(epoch_value_loss)

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # Initialize the DRL agent
    drl = DRL()

    # Initialize SummaryWriter
    writer = SummaryWriter(log_dir="runs/drl_experiment_0330_1")

    # Example: Log some scalar values (e.g., reward, loss, etc.)
    for episode in tqdm(range(100)):  # Assuming you run for 100 episodes
        drl.rollout(episode)  # Collect trajectory or interaction data
        reward = drl.get_avg_reward()  # Assume you have a method to get the current reward

        drl.update()

        policy_loss,value_loss = drl.get_avg_loss()  # Assume you have a method to get the loss

        # Log the reward and loss for each episode
        writer.add_scalar('Reward/episode', reward, episode)
        writer.add_scalar('Policy_Loss/episode', policy_loss, episode)
        writer.add_scalar('Value_Loss/episode', value_loss, episode)


    # Close the writer after training
    writer.close()
# if __name__=="__main__":

#     drl = DRL()
#     drl.rollout(0)
#     # print(drl.traj_data.states.shape)
#     drl.update()
