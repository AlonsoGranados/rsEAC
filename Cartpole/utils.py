import torch
import matplotlib.pyplot as plt
from itertools import count
from policies import greedy_action
from policies import select_action
import numpy as np
from tensordict import TensorDict

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

def evaluation(args, env_sampler, agent):
    returns = []
    initial_states = []
    for i in range(5):
        env_sampler.current_state = None
        env_sampler.path_length = 0
        done = False
        test_step = 0
        G = 0
        while (not done) and (test_step != args.max_path_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)

            if test_step == 0:
                with torch.no_grad():
                    state = torch.tensor(cur_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    V = torch.max(agent.policy_net(state)).item() #.argmax().cpu().tolist()
                    initial_states.append(V)

            G += reward
            test_step += 1
        returns.append(G)
    mean_return = np.mean(returns)
    mean_initial = np.mean(initial_states)
    return mean_return, mean_initial

def entropic(action_value_dist, agent):
    Z = action_value_dist * torch.exp((agent.V_RANGE - 100.0) / agent.beta)
    Z = torch.sum(Z, dim=2)
    return Z

def belief(args, env_sampler, agent):
    env_sampler.current_state = None
    done = False
    test_step = 0
    while (not done) and (test_step != args.max_path_length):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        if test_step % 100 == 0:
            x = torch.tensor(cur_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                plt.title(test_step)
                action_value_dist = agent.pred_net(x).squeeze(0)  # (N_ENVS, N_ACTIONS, N_ATOM)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[0, :].cpu().numpy(), label = 0)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[1, :].cpu().numpy(), label = 1)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[2, :].cpu().numpy(), label = 2)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[3, :].cpu().numpy(), label = 3)
                plt.legend()
                plt.show()
                z = entropic(action_value_dist, agent)
                print(test_step, z)
        if reward == -100.0:
            with torch.no_grad():
                plt.title('Horrible')
                action_value_dist = agent.pred_net(x).squeeze(0)  # (N_ENVS, N_ACTIONS, N_ATOM)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[0,:].cpu().numpy(), label = 0)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[1, :].cpu().numpy(), label = 1)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[2, :].cpu().numpy(), label = 2)
                plt.plot(agent.V_RANGE_numpy, action_value_dist[3, :].cpu().numpy(), label = 3)
                plt.legend()
                plt.show()

        test_step += 1


def plot_variational_variance(args, env_sampler, agent, prior_model):
    env_sampler.current_state = None
    env_sampler.path_length = 0
    done = False
    test_step = 0
    while (not done) and (test_step != args.max_path_length):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        action_one_hot_batch = np.zeros((4,1))
        action_one_hot_batch[action,0] = 1
        state_batch = cur_state.reshape((-1,1))
        next_state_batch = next_state.reshape((-1, 1))
        inputs = np.concatenate((state_batch, action_one_hot_batch, next_state_batch), axis=0)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=agent.device).squeeze()
        inputs = inputs.unsqueeze(0)
        mean_p, log_var_p = prior_model.model(inputs)
        var_p = torch.exp(log_var_p)
        # print()
        if next_state[0] > 0:
            x = plt.Circle((cur_state[0],cur_state[1]),np.sqrt(var_p.cpu().item())/10, color='r', fill=False)
        else:
            x = plt.Circle((cur_state[0], cur_state[1]), np.sqrt(var_p.cpu().item())/10, fill=False)
        # ax.add_patch(x)
        # plt.plot(cur_state[0], cur_state[1], 'o')
        plt.gca().add_patch(x)
        test_step += 1
    plt.ylim(0,3)
    plt.xlim(-3,3)
    plt.show()

def evaluate_Q(args, state, policy_net, device):
    i_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    if args.beta > 0:
        Q_I = torch.max(policy_net(i_state))
    else:
        Q_I = torch.min(policy_net(i_state))
    return Q_I.item()

def plot_action_values(policy_net, device):
    state = []
    for i in np.arange(-2,2.5,0.5):
        for j in np.arange(-0.2,0.21,0.1):
            state.append([i,0,j,0])
    state_batch = torch.FloatTensor(state).to(device)
    Q = policy_net(state_batch)
    Q = Q.cpu().detach().numpy()
    print(Q[0,0])
    plt.imshow(Q[:,0].reshape(-1,9), vmin=0,vmax=5)
    plt.show()

def plot_Z(args, memory, device, policy_net, target_net):
    state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(1)
    mask_batch_1 = mask_batch
    mask_batch = ~mask_batch

    state_batch = torch.FloatTensor(state_batch).to(device)
    action_batch = torch.IntTensor(action_batch).to(device).type(torch.int64).unsqueeze(1)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
    mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)
    mask_batch_1 = torch.FloatTensor(mask_batch_1).to(device).unsqueeze(1)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    if args.beta > 0:
        with torch.no_grad():
            min_qf_next_target = target_net(next_state_batch).max(1)[0]
            min_qf_next_target = min_qf_next_target.unsqueeze(1)
            min_qf_next_target = torch.clamp_min_(min_qf_next_target, 0)
            min_qf_next_target = torch.pow(min_qf_next_target, args.gamma)
    else:
        with torch.no_grad():
            min_qf_next_target = target_net(next_state_batch).min(1)[0]
            min_qf_next_target = min_qf_next_target.unsqueeze(1)
            min_qf_next_target = torch.clamp_min_(min_qf_next_target, 0.0005)
            min_qf_next_target = torch.pow(min_qf_next_target, args.gamma)
    # print(target_net(next_state_batch))
    print('Run')
    print(state_action_values.item(), torch.exp(reward_batch / args.beta).item(), (mask_batch_1 + mask_batch * min_qf_next_target).item(), mask_batch_1.item(),mask_batch.item())
    print(policy_net(state_batch))

# from DQN_network import DQN
# device = 'cuda'
# policy_net = DQN(4, 2).to(device)
# plot_action_values(policy_net, device)