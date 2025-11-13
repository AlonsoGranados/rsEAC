import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)
# plt.rcParams['text.usetex'] = True
class Clamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=-3, max=3)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class AdaptedL1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        TD = torch.clamp(input - target, min=-5, max=5)
        clamped_target = torch.clamp(target, min=-5, max=5)
        return torch.exp(clamped_target)*torch.abs(torch.exp(TD) -1)

    @staticmethod
    def backward(ctx, grad_output):
        input, target,  = ctx.saved_tensors
        clamped_input = torch.clamp(input, min=-4, max=4)
        grad_input = grad_output * torch.exp(clamped_input) * torch.sign(input-target)
        # grad_input = grad_output * torch.sign(input - target)
        return grad_input, None

class AdaptedL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        TD = torch.clamp(input - target, min=-4, max=4)
        clamped_target = torch.clamp(target, min=-4, max=4)
        return torch.exp(2*clamped_target)*torch.square(torch.exp(TD) -1)

    @staticmethod
    def backward(ctx, grad_output):

        input, target, = ctx.saved_tensors
        l = torch.maximum(2*input, input+target)
        z = torch.max(l)

        e = np.random.random()
        i = 0
        if e < 0.0001:
            den = False
            i += 1
            print(z)
            a = l.detach().cpu().numpy()
            plt.hist(a, bins=20, label=r'$m_\beta^i$', density=den, alpha=0.8)
            # plt.savefig('histogram{0}.pdf'.format(i), bbox_inches='tight')
            # plt.show()

            a = a - z.item()
            plt.hist(a, bins=20, label = r'$m_\beta^i - z$', density=den, alpha=0.8)
            plt.legend()
            # plt.savefig('histogram_shift{0}.pdf'.format(i), bbox_inches='tight')
            # plt.show()

            a = np.clip(a, -5, 5)
            plt.hist(a, bins=10, label = r'clip$(m_\beta^i - z, -5, 5)$', density=den, alpha=0.8)
            plt.legend()
            plt.ylabel('Counts')
            # plt.xlabel(r'e^{clip(m_\beta^i -z, -5,5)}')
            plt.xlabel('Values')
            plt.savefig('histogram_all{0}.pdf'.format(i), bbox_inches='tight')
            plt.show()
        first_term = torch.exp(torch.clamp(l - z, min= -5, max = 5))
        second_term = 1 - torch.exp(-torch.abs(input-target) )
        grad_input = grad_output * first_term * second_term * torch.sign(input-target)
        return grad_input, None


class Agent():
    def __init__(self, policy_net, target_net, optimizer, device, env, beta):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.device = device
        self.epsilon = 0.1
        self.env = env
        self.beta = beta

    def sample_tuples(self, memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(256)

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        mask_batch = torch.tensor(~done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch, done_batch

    def select_action(self, state, eval=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if eval == False:
            sample = random.random()
            if sample > self.epsilon:
                with torch.no_grad():
                    if self.beta > 0:
                        action = self.policy_net(state).argmax().cpu().tolist()
                    else:
                        action = self.policy_net(state).argmin().cpu().tolist()
            else:
                action = self.env.action_space.sample()
        else:
            if self.beta > 0:
                action = self.policy_net(state).argmax().cpu().tolist()
            else:
                action = self.policy_net(state).argmin().cpu().tolist()
        # if self.beta > 0:
        #
        # else:
        #     state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        #     if eval == False:
        #         sample = random.random()
        #         if sample > self.epsilon:
        #             with torch.no_grad():
        #                 action = self.policy_net(state).argmin().cpu().tolist()
        #         else:
        #             action = self.env.action_space.sample()
        #     else:
        #         action = self.policy_net(state).argmin().cpu().tolist()
        return action

    def soft_update(self, args):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (
                        1 - args.tau)
        self.target_net.load_state_dict(target_net_state_dict)

class DQN_agent(Agent):
    def __init__(self, policy_net, target_net, optimizer, device, env, beta):
        super().__init__(policy_net, target_net, optimizer, device, env, beta)

    def optimize_model(self, args, memory, reward_model):
        if len(memory) < args.batch_size:
            return

        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = self.sample_tuples(memory)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(256)

        # encoded_arr = np.zeros((action_batch.shape[0], 4))
        # encoded_arr[np.arange(action_batch.shape[0]), action_batch] = 1
        #
        # inputs = np.concatenate((state_batch, encoded_arr, next_state_batch), axis=-1)

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        mask_batch = torch.tensor(~done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        # inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        #
        # with torch.no_grad():
        #     mean, var = reward_model.variational_model(inputs)
        #     reward_batch = torch.normal(mean, torch.sqrt(var))

        qf1 = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            qf1_next_target = self.target_net(next_state_batch).max(1)[0]
            qf1_next_target = qf1_next_target.unsqueeze(1)
            next_q_value = reward_batch + mask_batch * args.gamma * qf1_next_target

        # Compute Huber loss
        # criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        loss = criterion(qf1, next_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        self.soft_update(args)

        return loss.item()

class DZN_agent(Agent):
    def __init__(self, policy_net, target_net, optimizer, device, env):
        super().__init__(policy_net, target_net, optimizer, device, env)

    def optimize_model(self, args, memory):
        if len(memory) < args.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, done_batch = self.sample_tuples(memory)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        if args.beta > 0:
            with torch.no_grad():
                qf_next_target = self.target_net(next_state_batch).max(1)[0]
        else:
            with torch.no_grad():
                qf_next_target = self.target_net(next_state_batch).min(1)[0]

        qf_next_target = qf_next_target.unsqueeze(1)
        qf_next_target = torch.clamp_min_(qf_next_target, 0.00001)
        qf_next_target = torch.pow(qf_next_target, args.gamma)

        next_q_value = torch.exp(reward_batch / args.beta) * (done_batch + mask_batch * qf_next_target)

        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, next_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # Soft update of the target network's weights
        self.soft_update(args)

        return loss.item()

class stable_DZN_agent(Agent):
    def __init__(self, policy_net, target_net, optimizer, device, env, beta):
        super().__init__(policy_net, target_net, optimizer, device, env, beta)
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer

    def optimize_model(self, args, memory):
        if len(memory) < args.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = self.sample_tuples(memory)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        if args.beta > 0:
            with torch.no_grad():
                qf_next_target = self.target_net(next_state_batch).max(1)[0]
        else:
            with torch.no_grad():
                qf_next_target = self.target_net(next_state_batch).min(1)[0]

        qf_next_target = qf_next_target.unsqueeze(1)
        next_q_value = reward_batch / args.beta + (mask_batch * qf_next_target * args.gamma)

        # adapt_class = AdaptedL1()
        adapt_class = AdaptedL2()
        loss = adapt_class.apply(state_action_values, next_q_value)
        loss = torch.mean(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        total_norm = 0
        for p in self.policy_net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        self.soft_update(args)

        return loss.item(), total_norm
