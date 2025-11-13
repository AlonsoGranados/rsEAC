import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim)
        )

        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.max_action * torch.tanh(x)
        return mu

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def Q1(self, state, action):
        x = torch.cat((state, action), 1)
        q1 = self.q1_net(x)
        return q1

class AdaptedL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        TD = torch.clamp(input - target, min=-4, max=4)
        clamped_target = torch.clamp(target, min=-4, max=4)
        return torch.exp(2*clamped_target)*torch.square(torch.exp(TD) -1)
        # return torch.square(input - target)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, = ctx.saved_tensors

        l = torch.maximum(2 * input, target + input)
        z = torch.max(l)
        # first_term = torch.exp(l - z + 1)
        # e = np.random.random()
        # if e < 0.0001:
        #     print(z)
        #     a = l.detach().cpu().numpy()
        #     plt.hist(a, bins='auto')
        #     plt.show()

        # l = torch.maximum(2 * input, target + input)
        # z = torch.min(l)

        first_term = torch.exp(torch.clamp(l - z, min=-5, max=5))
        # first_term = torch.exp(torch.clamp(l - z + 1, max= 4))
        second_term = 1 - torch.exp(-torch.abs(input - target))


        # e = np.random.random()
        # if e < 0.000005:
        #     print(z)
        #     a = l.detach().cpu().numpy()
        #     plt.title('2Q and Q+R+Q')
        #     plt.hist(a, bins='auto')
        #     plt.show()
        #     b = second_term.detach().cpu().numpy()
        #     plt.title('1-e^k')
        #     plt.hist(b, bins='auto')
        #     plt.show()

        # second_term = second_term / torch.max(second_term)

        grad_input = first_term * second_term * torch.sign(input - target)
        return grad_input, None

class TD3Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            net_width,
            max_action,
            lr,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            beta = 10.0,
            save_dir = None,
            algo = 'standard'
    ):
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.beta = beta
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.save_dir = save_dir
        self.algo = algo

        self.total_it = 0
        self.online_rewards = deque(maxlen=int(1e4))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            if self.beta > 0:
                target_Q = torch.min(target_Q1, target_Q2)
            else:
                target_Q = torch.max(target_Q1, target_Q2)

            target_Q = reward/self.beta + not_done * target_Q * self.discount
            # target_Q = torch.exp(reward/-20.0) * (1 - not_done + torch.pow(torch.clamp(target_Q, 0.1), 0.9) * not_done)
            # target_Q = torch.clamp(target_Q,0.1)
            # print(torch.mean(target_Q))
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        if self.algo == 'standard':
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            adapt_class1 = AdaptedL2()
            adapt_class2 = AdaptedL2()
            qf1_loss = torch.mean(adapt_class1.apply(current_Q1, target_Q))
            qf2_loss = torch.mean(adapt_class2.apply(current_Q2, target_Q))

            # Compute critic loss
            critic_loss = qf1_loss + qf2_loss
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            if self.beta > 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            else:
                actor_loss = self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, n_stp):
        save_critic_name = self.save_dir + '/stp_{}_critic.pkl'.format(n_stp)
        save_critic_opt_name = self.save_dir + '/stp_{}_critic_opt.pkl'.format(n_stp)
        torch.save(self.critic.state_dict(), save_critic_name)
        torch.save(self.critic_optimizer.state_dict(), save_critic_opt_name)
        
        save_actor_name = self.save_dir + '/stp_{}_actor.pkl'.format(n_stp)
        save_actor_opt_name = self.save_dir + '/stp_{}_actor_opt.pkl'.format(n_stp)
        torch.save(self.actor.state_dict(), save_actor_name)
        torch.save(self.actor_optimizer.state_dict(), save_actor_opt_name)

    def save_best(self, risk=True):
        if risk:
            save_critic_name = self.save_dir + '/best_critic_var.pkl'
            torch.save(self.critic.state_dict(), save_critic_name)
            save_actor_name = self.save_dir + '/best_actor_var.pkl'
            torch.save(self.actor.state_dict(), save_actor_name)
        else:
            save_critic_name = self.save_dir + '/best_critic_mean.pkl'
            torch.save(self.critic.state_dict(), save_critic_name)
            save_actor_name = self.save_dir + '/best_actor_mean.pkl'
            torch.save(self.actor.state_dict(), save_actor_name)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

