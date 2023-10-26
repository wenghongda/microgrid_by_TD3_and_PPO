from model import Actor,Critic
import torch.optim as optim
from parameters import *
import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0')
torch.cuda.empty_cache()

class PPO:
    def __init__(self,N_S,N_A):
        self.actor_net = Actor(N_S,N_A)
        self.actor_old_net = Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=lr_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self,memory):
        memory = np.array(memory,dtype="object")
        states = torch.tensor(np.vstack(memory[:,0]),dtype=torch.float32)

        # the shape of states are (length of memory,1)
        actions = torch.tensor(np.vstack(memory[:,1]),dtype=torch.float32)
        rewards = torch.tensor(np.vstack(memory[:,2]),dtype=torch.float32)
        masks = torch.tensor(np.vstack(memory[:,3]),dtype=torch.float32)
        values = self.critic_net(states)
        # Normalizing the rewards
        #rewards = torch.tensor(rewards,dtype=torch.float32)
        #rewards = (rewards-rewards.mean())/(rewards.std() + 1e-7)

        returns,advants = self.get_gae(rewards,masks,values)
        self.actor_old_net.load_state_dict(self.actor_net.state_dict())
        with torch.no_grad():
            old_mu,old_std = self.actor_old_net(states)
            pi = self.actor_old_net.distribution(old_mu,old_std)
            old_log_prob = pi.log_prob(actions).sum(dim=1,keepdims=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(5):
            #np.random.shuffle(arr)
            for i in range(n//24):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index] # the shape of b_states (batch_size,1)
                b_advants = advants[b_index] # b_advants.shape:(batch_size,1)
                b_actions = actions[b_index]              # b_actions.shape:(batch_size,)

                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(dim =1,keepdims=True)
                old_prob = old_log_prob[b_index].detach()

                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants.squeeze(1)

                values = self.critic_net(b_states)
                critic_loss = self.critic_loss_func(values,returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)
                clipped_loss = ratio*b_advants.squeeze(1)
                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
            self.actor_old_net.load_state_dict(self.actor_net.state_dict())

    #calculate TD_target
    def get_td_target(self,rewards,masks,values):
        previous_value = 0
        td_target=torch.zeros_like(rewards)
        for t in reversed(range(0,len(rewards))):
            running_td_target= rewards[t] + gamma*previous_value*masks[t]
            previous_value = values.data[t]
            td_target[t] = running_td_target
        return td_target
    #calculate GAE
    def get_gae(self,rewards,masks,values):
        rewards = torch.as_tensor(rewards)
        masks = torch.as_tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0,len(rewards))):
            running_returns = rewards[t] + (gamma * running_returns)*masks[t]
            running_tderror = (rewards[t] + gamma*previous_value)*masks[t] - \
                              values.data[t]
            running_advants = running_tderror + gamma * lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        advants = (advants - advants.mean()) / (advants.std() + 1e-7)
        return returns, advants
    def save(self,checkpoint_path):
        torch.save(self.actor_net.state_dict(),checkpoint_path)

    def load(self,checkpoint_path):
        self.critic_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))