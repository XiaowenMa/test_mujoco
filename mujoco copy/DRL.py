import torch
from PPOAgent import PPOAgent
from torch.optim import Adam
import gymnasium as gym
from customEnv import MyEnv
from TrajData import TrajData
# from ReplayBuffer import ReplayBuffer

class DRL:
    def __init__(self):

        self.n_envs = 4 # for testing
        self.n_steps = 500 # about 1 sim sec
        # TODO: check model's qpos+qvel
        self.n_obs = 35+34 # model.qvel+model.qpos for now, suppose we only observe pos and vel
        self.n_actions = 28 # 28 actuators, each action modeled as a gaussian

        # self.envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")  for _ in range(self.n_envs)])
        self.envs = gym.vector.SyncVectorEnv([lambda: MyEnv("test.xml") for _ in range(self.n_envs)])

        # self.traj_datas = []
        # self.replay_buffer = ReplayBuffer()
        self.traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions, 0,0) # placeholder for ind, maybe no longer in use
  
        self.agent = PPOAgent(self.n_obs, n_actions=self.n_actions, a_lambda=.95, gamma = .99)  
        self.optimizer = Adam(self.agent.parameters(), lr=1e-3)
        # self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i):
        # traj_ind = len(self.traj_datas)
        # traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions,traj_ind)
        obs, _ = self.envs.reset() # obs, reset_info
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            # print(actions.shape)
            log_probs = probs.log_prob(actions).sum(-1)

            # for i in range(5): #frame_skip = 5
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            # self.replay_buffer.store(obs,actions,rewards,next_obs,done,traj_ind,t)
            obs = torch.Tensor(next_obs)
                

        self.traj_data.calc_returns()
        # self.traj_datas.append(traj_data)
        print("avg reward: ", self.traj_data.rewards.mean())

        # self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        # self.writer.flush()


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 10 if self.agent.name == 'PPO' else 1

        for _ in range(epochs):

            # # sample batch from replay buffer
            # samples = self.replay_buffer.sample()

            # # for each sample, update value network and policy network
            # for state, action, reward, next_state, done, traj_ind, t in samples:
            #     traj_data = self.traj_datas[traj_ind]
            #     loss = self.agent.get_loss(traj_data)


            loss = self.agent.get_loss(self.traj_data)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.traj_data.detach()
if __name__=="__main__":
    drl = DRL()
    drl.rollout(0)
    # print(drl.traj_data.states.shape)
    drl.update()
