# Library Imports
import numpy as np
import random
import torch
import pandas as pd
import torch.nn as nn
from torch import optim as optim
from parameters import *
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from collections import deque
from energy_system import EnergySystem
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2021)


class Critic(nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float = None, density: int = 256, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.Q = torch.nn.Linear(density, 1, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        #self.set_init([self.H1, self.H2, self.H3, self.Q])
        self.device = device
        self.to(device)
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action),dim = 1)
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.Q(value)
        return value


class Actor(nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float = None, density: int = 128, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.mu = torch.nn.Linear(density, n_actions, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = F.relu(self.H1(state))
        action = F.relu(self.H2(action))
        action = torch.tanh(self.mu(action))

        return action

class TD3:
    def __init__(self,env,n_games: int = 5, training: bool = True,
                 alpha=0.0003, beta=0.0002, gamma=0.95, tau=0.004,
                 batch_size: int = 64, noise: str = 'normal'):
        """Initialize an Agent object.
        Params
        =====
            alpha & beta :learning rate
            n_actions(int):dimension of each action
            obs_shape(int):dimension of each state
            n_games(int): the total amount of iterations
            gamma(float):discount rate
            buffer_len(int):the length of Replaybuffer
            seed(int):random seed to evaluate the training effect
            per_alpha(int),per_beta(int):per_alpha and per_beta are hypeparameters to represent
            how often PER(prioritiy experience replay) is used to train the model. In general, alpha corresponds with beta
            tau (float) : the update degree of target_network and original_network
        """
        self.directory = r'D:\whd_disertation\new_model_dict\TD3\\'
        self.reward_directory = r"D:\whd_disertation\record\TD3\new\training_reward.csv"
        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_actions: int = N_A

        self.obs_shape: int = N_S
        self.n_games = n_games
        self.gamma = gamma
        self.is_training = training
        self.memory = deque(maxlen=100_000)

        self.batch_size = batch_size
        self.noise = noise
        self.noise_clip = 1
        #self.max_action = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.load = False
        self.policy_delay = 5

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor').to(device)
        self.actor_target = Actor(self.obs_shape,self.n_actions,alpha,name='actor_target').to(device)

        self.critic_1 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_1_target = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2 = Critic(self.obs_shape+self.n_actions,beta).to(device)
        self.critic_2_target = Critic(self.obs_shape+self.n_actions,beta).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.zeros_like(action)
        std = 0.2*torch.ones_like(action)
        noise = torch.normal(mean,std).clamp(-self.noise_clip,self.noise_clip).to("cuda")

        return noise
    def sample(self,batch_size):

        memory_for_training = random.sample(self.memory,batch_size)

        return memory_for_training

    def choose_action(self, observation: torch.tensor) -> np.ndarray:
        self.actor.eval()

        state = torch.as_tensor(observation, dtype=torch.float32, device=device)
        action = self.actor.forward(state)
        action = torch.as_tensor(action,dtype=torch.float32,device=device)
        if self.is_training:
            action += self._add_exploration_noise(action)
        return action.detach().cpu().numpy()

    def train(self):
        gamma = self.gamma
        replayer_buffer = self.memory
        batch_size = self.batch_size
        #Sample replay buffer
        for i in range(self.n_games):
            memory_total = np.array(replayer_buffer,dtype='object')
            reward_total = torch.tensor(np.vstack(memory_total[:, 3]), dtype=torch.float32).to(device)
            idx_for_training = random.sample(range(0,len(memory_total)),batch_size)
            memory_for_training = memory_total[idx_for_training]
            state = torch.tensor(np.vstack(memory_for_training[:, 0]), dtype=torch.float32).to(device)

            # the shape of states are (length of memory,1)
            action = torch.tensor(np.vstack(memory_for_training[:, 1]), dtype=torch.float32).to(device)
            next_state = torch.tensor(np.vstack(memory_for_training[:,2]),dtype=torch.float32).to("cuda")
            #rewards = reward_total[idx_for_training:idx_for_training+batch_size]
            rewards = torch.tensor(np.vstack(memory_for_training[:, 3]), dtype=torch.float32).to(device)

            done = torch.tensor(np.vstack(memory_for_training[:, 4]), dtype=torch.float32).to(device)
            #Select next action through target_actor network
            noise = self._add_exploration_noise(action)
            next_action = self.actor_target(next_state)+ noise
            min_action = torch.tensor([-1,-1,-1],dtype=torch.float32).to(device)
            max_action = torch.tensor([1,1,1],dtype=torch.float32).to(device)

            next_action = torch.clamp(next_action,min_action,max_action)
            #compute target q_value：
            target_q1 = self.critic_1_target(next_state,next_action)
            target_q2 = self.critic_2_target(next_state,next_action)
            target_q = torch.min(target_q1,target_q2)
            target_q = rewards + ((1 - done) * gamma * target_q).detach()
            # Optimize Critic 1:
            current_q1 = self.critic_1(state,action)
            loss_q1 = F.mse_loss(current_q1,target_q)
            self.critic_1.optimizer.zero_grad()
            loss_q1.backward()
            self.critic_1.optimizer.step()
            #self.writer.add_scalar('Loss/Q1_loss', loss_q1, global_step=self.num_critic_update_iteration)
            # Optimize Critic 2:
            current_q2 = self.critic_2(state,action)
            loss_q2 = F.mse_loss(current_q2,target_q)

            self.critic_2.zero_grad()
            loss_q2.backward()
            self.critic_2.optimizer.step()

            #Delayed policy updates:
            if (i+1) % self.policy_delay == 0:
                #Compute actor loss:
                actor_loss = - self.critic_1(state,self.actor(state)).mean()

                #Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                #self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                    target_param.data.copy_((1 - self.tau)* target_param.data +self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(),self.critic_2_target.parameters()):
                    target_param.data.copy_((1-self.tau) * target_param + self.tau*param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
        self.save_models()
        print("No. {} training episode".format(self.num_training))

    def save_models(self):
        torch.save(self.actor.state_dict(),self.directory + 'actor.pth')
        torch.save(self.actor_target.state_dict(),self.directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(),self.directory + 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(),self.directory + 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(),self.directory + 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(),self.directory + 'critic_2_target.pth')
        self.load = True
        print("==================================================")
        print("Model has been saved...")
        print("==================================================")

    def load_models(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.directory + 'critic_2_target.pth'))
        print("==================================================")
        print("Model has been loaded...")
        print("==================================================")

if __name__ == '__main__':
    env = EnergySystem(0.5,0.5,5,150,-10,'train')
    agent = TD3(env)
    #if agent.load:
    #   agent.load_models()
    #Warm up stage and dont update
    state = env.reset()
    state = torch.tensor(state,dtype=torch.float32)
    # Warm up
    for t in range(192*3):
        action = agent.choose_action(state)
        noise = np.random.normal(0,0.03,size=len(action))
        action = action + noise
        action = action.clip(np.array([-1,-1,-1],dtype=np.float32),np.array([1,1,1],dtype=np.float32))
        next_state,reward,done,_,_,_ = env.step(action)
        next_action = agent.choose_action(next_state)

        state = torch.as_tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
        action = torch.as_tensor(action,dtype=torch.float32,device=device).unsqueeze(0)
        next_state = torch.as_tensor(next_state,dtype=torch.float32,device=device).unsqueeze(0)
        next_action =torch.as_tensor(next_action,dtype=torch.float32,device=device).unsqueeze(0)

        q_state_1 = agent.critic_1.forward(state,action)
        q_state_2 = agent.critic_2(state,action)
        q_next_state_1 = agent.critic_1_target(next_state,next_action)
        q_next_state_2 = agent.critic_2_target(next_state,next_action)
        td_target = reward + (1-done) * min(q_next_state_1,q_next_state_2)
        td_error_1 = q_state_1 - td_target
        td_error_2 = q_state_2 - td_target
        td_error = (td_error_1+ td_error_2)/2

        state = state.squeeze(0).cpu().numpy()
        action = action.squeeze(0).cpu().numpy()
        td_error = td_error.squeeze(0)
        next_state = next_state.squeeze(0).cpu().numpy()

        agent.memory.append([state,action,next_state,reward,done])
        state = next_state
        if done == 1:
            env.reset()
        if (t+1) % 10 == 0:
            print('No. {} Warm up episode'.format(t+1))

    #Start training
    episode_history = []
    episode_reward_history = []
    for i in range(300):
        state = env.reset()
        epsiode_reward = 0
        total_cost = 0
        for t in range(192):

            action = agent.choose_action(state)
            noise = np.random.normal(0, 0.03, size=len(action))
            action = action + noise
            action = action.clip(np.array([-1,-1,-1],dtype=np.float32),np.array([1,1,1],dtype=np.float32))
            next_state, reward, done,_,_,cost = env.step(action)
            state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            state = state.squeeze(0).cpu().numpy()
            action = action.squeeze(0).cpu().numpy()
            next_state = next_state.squeeze(0).cpu().numpy()
            agent.memory.append([state, action, next_state, reward, done])
            state = next_state
            epsiode_reward += reward
            total_cost += cost
            if done == 1:
                env.reset()
            if (t+1) % 64 == 0:
                agent.train()
            print("No.{} episode and No.{} time step".format(i,t))
            if (t+1) % 24 == 0:
                print("the real total cost is :{}".format(total_cost))
                total_cost = 0
        print("current epsiode_reward is {}".format(epsiode_reward))
        episode_history.append(i+1)
        episode_reward_history.append(epsiode_reward)
        if (i+1) % 50 == 0:
            plt.plot(episode_history,episode_reward_history)
            plt.show()

        agent.save_models()
    df = pd.read_csv(agent.reward_directory)
    df['reward'] = episode_reward_history
    df.to_csv(agent.reward_directory,index=False)