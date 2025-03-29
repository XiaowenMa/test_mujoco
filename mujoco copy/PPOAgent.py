import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions import Normal
from TrajData import TrajData

class PPOAgent(nn.Module):
    def __init__(self, n_obs, n_actions, a_lambda, gamma=.99, epochs=10): # for this model, ? actuator
        super().__init__()
        self.name = 'PPO'
        self.epochs = epochs

        self.policy = nn.Sequential(
            nn.Linear(n_obs,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(n_obs,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
        # TODO: Same optimizer here, original paper used 2 optimizers

        self.a_lambda = a_lambda
        self.gamma = gamma

        # end student code

    """
        Version 1: Use entire traj as in HW1, in the original paper, in each update step, the authors sampled from memory(traj?)
    """
    def get_loss(self, traj_data:TrajData, epsilon=.2):

        # # todo: student code here
        # policy_loss = []
        # value_loss = []
        # T = traj_data.n_steps
        # # print(traj_data.not_dones.shape,traj_data.not_dones[-1])

        # advantages = torch.zeros_like(traj_data.rewards)
        # values = torch.zeros_like(traj_data.rewards)
        # values[-1] = self.value(traj_data.states[-1]).flatten()
        # # calc advantages
        # gae = torch.zeros_like(values[-1])
        # for t in range(T-2,-1,-1):
        #   value = self.value(traj_data.states[t]).flatten()
        #   values[t] = value

        #   next_value = values[t+1]*(traj_data.not_dones[t])
        #   delta = traj_data.rewards[t]+self.gamma*next_value-value
        #   # print(traj_data.not_dones[t])
        #   gae = delta + self.gamma*self.a_lambda*traj_data.not_dones[t]*gae
        #   advantages[t] = gae


        # for t in range(T):
        #   # A_gae = 0
        #   value_loss.append((traj_data.returns[t]-values[t])**2) # need to calc value_loss before t_prime loop

        #   actions,probs = self.get_action(traj_data.states[t])
        #   # print(actions.shape)
        #   p = probs.log_prob(traj_data.actions[t]).sum(-1)
        # #   print("P: ", p.shape)
        #   ratio = torch.exp(p-traj_data.log_probs[t])
        #   policy_loss.append(torch.min(ratio*advantages[t],self.clip(ratio,epsilon)*advantages[t]))
        # # print(policy_loss[1].shape,policy_loss[-1].shape)
        # policy_loss = -torch.stack(policy_loss).mean()
        # value_loss = torch.stack(value_loss).mean()

        # loss = policy_loss+value_loss  # replace

        # print(f"policy loss: {policy_loss.item()}, value loss: {value_loss.item()}")
        # return loss
    

            # todo: student code here
        policy_loss = []
        value_loss = []
        T = traj_data.n_steps
        # print(traj_data.not_dones.shape,traj_data.not_dones[-1])

        advantages = torch.zeros_like(traj_data.rewards)
        values = torch.zeros_like(traj_data.rewards)
        values[-1] = self.value(traj_data.states[-1]).flatten()
        # calc advantages
        gae = torch.zeros_like(values[-1])
        for t in range(T-2,-1,-1):
          value = self.value(traj_data.states[t]).flatten()
          values[t] = value

          next_value = values[t+1]*(traj_data.not_dones[t])
          delta = traj_data.rewards[t]+self.gamma*next_value-value
          # print(traj_data.not_dones[t])
          gae = delta + self.gamma*self.a_lambda*traj_data.not_dones[t]*gae
          advantages[t] = gae


        for t in range(T):
          # A_gae = 0
          value_loss.append((traj_data.returns[t]-values[t])**2) # need to calc value_loss before t_prime loop

          actions,probs = self.get_action(traj_data.states[t])
          # print(actions.shape)
          p = probs.log_prob(traj_data.actions[t]).sum(-1)
          # print("P: ", p.shape)
          ratio = torch.exp(p-traj_data.log_probs[t])
          policy_loss.append(torch.min(ratio*advantages[t].detach(),self.clip(ratio,epsilon)*advantages[t].detach()))
        # print(policy_loss[1].shape,policy_loss[-1].shape)
        policy_loss = -torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()
        # print(policy_loss,value_loss)

        loss = policy_loss+value_loss  # replace
        # print(loss)
        # end student code
        print(value_loss.item())
        return policy_loss,value_loss


    def clip(self,ratio, epsilon):
        return torch.clamp(ratio, 1-epsilon,1+epsilon)

    def get_action(self, obs):

        """TODO: update the clamp part, clamp the log_std_dev instead of std_dev"""
        # mean, log_std_dev = self.policy(obs).chunk(2, dim=-1)
        # mean = torch.tanh(mean)
        # mean = -50+100*(mean+1)/2 # suppose torque -50 - 50

        # std_dev = log_std_dev.exp().clamp(.2, 2)
        # dist = Normal(mean,std_dev)
        # action = dist.rsample()
        mean= self.policy(obs)
        std_dev = torch.ones_like(mean)

        dist = Normal(mean,std_dev)
        action = dist.rsample()

        return action,dist