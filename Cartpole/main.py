import gymnasium as gym
import argparse

import torch
import torch.optim as optim
import math
from replay_memory import ReplayMemory
from sample_env import EnvSampler
from Algorithms.DQN import DQN_algorithm
from Algorithms.DZN import DZN_algorithm
from DZN_network import stable_DZN
from Agent import stable_DZN_agent
from utils import evaluation
import itertools
from utils import belief
import numpy as np
import matplotlib.pyplot as plt

def readParser():
    parser = argparse.ArgumentParser(description='DZN')
    parser.add_argument('--env_name', default="CartPole-v1",
                        help='Environment')
    parser.add_argument('--agent', default="stable_DZN",
                        help='Network')
    parser.add_argument('--gamma', default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--experiment_num', type=int, default = 6,
                        help='experiment number')
    parser.add_argument('--tau', default=0.005,
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--beta', default=10.0,
                        help='Risk parameter')
    parser.add_argument('--lr', default=0.0001,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--experience_replay_size', default=500000,
                        help='size of experience buffer')
    parser.add_argument('--epoch_length', default=1000,
                        help='steps per epoch')
    parser.add_argument('--batch_size', default=128,
                        help='batch size for training policy')
    parser.add_argument('--num_episodes', default=30,
                        help='number of episodes per epoch')
    parser.add_argument('--max_path_length', default=500,
                        help='number of episodes per epoch')
    parser.add_argument('--init_exploration_steps', default=10000,
                        help='number of episodes per epoch')
    parser.add_argument('--num_epoch', default=200,
                        help='total number of epochs')
    return parser.parse_args()

def experiment(args, env_sampler, trainer, memory):
    G = []
    initial_values = []
    gradient_norms = []
    EPS_END = 0.1
    #Collect initial data
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.exploratory_sample()
        memory.push(cur_state, action, reward, next_state, done)
    env_sampler.current_state = None
    env_sampler.path_length = 0
    print('Done')
    #Training
    for epochs in range(args.num_epoch):
        trainer.epsilon = max(EPS_END, trainer.epsilon * 0.9)
        for i in range(args.epoch_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(trainer)
            memory.push(cur_state, action, reward, next_state, done)
            loss, gradients = trainer.optimize_model(args, memory)

        mean_return, mean_initial = evaluation(args, env_sampler, trainer)
        G.append(mean_return)
        initial_values.append(mean_initial)
        gradient_norms.append(gradients)

        print(epochs, mean_return, mean_initial, gradients)
        if (epochs + 1) % 10 == 0:
            np.save('./Experiments/{0}/{1}/return_{2}_{3}'.format(args.env_name, args.agent, int(args.experiment_num),
                                                           args.beta), G)
            np.save('./Experiments/{0}/{1}/initial_{2}_{3}'.format(args.env_name, args.agent, int(args.experiment_num),
                                                                  args.beta), initial_values)
            np.save('./Experiments/{0}/{1}/gradient_{2}_{3}'.format(args.env_name, args.agent, int(args.experiment_num),
                                                                  args.beta), gradient_norms)
    print('Complete')

def main(args = None):
    if args is None:
        args = readParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    env = gym.make(args.env_name)

    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    memory = ReplayMemory(10000)

    if args.agent == 'DQN':
        trainer = DQN_algorithm(args, n_observations, n_actions, device, env)
    elif args.agent == 'DZN':
        trainer = DZN_algorithm(args, n_observations, n_actions, device, env)
    elif args.agent == 'stable_DZN':
        policy_net = stable_DZN(n_observations, n_actions).to(device)
        target_net = stable_DZN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
        trainer = stable_DZN_agent(policy_net, target_net, optimizer, device, env, args.beta)
    else:
        return

    experiment(args, env_sampler, trainer, memory)

if __name__ == '__main__':
    main()
