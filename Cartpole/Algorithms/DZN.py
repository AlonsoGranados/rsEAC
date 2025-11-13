import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

class Z_network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Z_network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        # return torch.exp(self.layer3(x))

class DZN_algorithm():
    def __init__(self, args, n_observations, n_actions, device, env):
        self.policy_net = Z_network(n_observations, n_actions).to(device)
        self.target_net = Z_network(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)
        self.device = device
        self.epsilon = 0.1
        self.env = env
        self.beta = args.beta

    def select_action(self, state, eval=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not eval:
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
        return action

    def soft_update(self, args):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (
                        1 - args.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self, args, memory):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(256)

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        mask_batch = torch.tensor(~done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        qf1 = self.policy_net(state_batch).gather(1, action_batch)

        # with torch.no_grad():
        #     qf1_next_target = self.target_net(next_state_batch).max(1)[0]
        #     qf1_next_target = qf1_next_target.unsqueeze(1)
            # next_q_value = reward_batch + mask_batch * args.gamma * qf1_next_target

        with torch.no_grad():
            if args.beta > 0:
                qf_next_target = self.target_net(next_state_batch).max(1)[0]
            else:
                qf_next_target = self.target_net(next_state_batch).min(1)[0]

            qf_next_target = qf_next_target.unsqueeze(1)
            qf_next_target = torch.clamp_min_(qf_next_target, 0.000001)
            qf_next_target = torch.pow(qf_next_target, args.gamma)

            next_q_value = torch.exp(reward_batch / args.beta) * (done_batch + mask_batch * qf_next_target)

        # Compute Huber loss
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        loss = criterion(qf1, next_q_value)

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
