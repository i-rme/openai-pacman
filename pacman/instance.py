import numpy as np
import gym
import sys
import pylab
import random

from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers

from pacman.core.deep_Q import DeepQAgent
from pacman.core.duel_Q import DuelQAgent

# Constants
EPISODES                = 4
ENVIROMENT              = 'MsPacman-ram-v0'
DEFAULT_TRAINING_PATH   = './results/'

class PacMan:
    def __init__(self, network, mode, view):
        env_to_wrap = gym.make(ENVIROMENT)
        self.env = wrappers.Monitor(env_to_wrap, './video', force=True)
        self.env.reset()

        print('\033[95m' + 'INFO: Using', network, 'on',ENVIROMENT + '\033[0m')

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        # Construct appropiate network based on flags
        if mode.lower() == 'test':
            load_model = True
        else:
            load_model = False

        if network == 'DDQN':
            self.agent = DeepQAgent(state_size, action_size, load_model)
        elif network == 'DQN':
            self.agent = DuelQAgent(self)

        if view:
            # Rendering MsPacman
            print('\033[95m' + 'INFO: Render is enabled'+ '\033[0m')
            self.agent.render = True
        else:
            print('\033[95m' + 'INFO: Render is disabled'+ '\033[0m')
            self.agent.render = False

    def train(self, path, statistics, mode):
        print('\033[95m' + 'INFO: Running' + '\033[0m')

        if path:
            TRAINING_PATH = path
        else:
            TRAINING_PATH = DEFAULT_TRAINING_PATH

        print('\033[95m' + 'INFO: Path set to', TRAINING_PATH + '\033[0m')

        if statistics:
            print('\033[95m' + 'INFO: Statistics are on, scores will be plotted'+'\033[0m')
        else:
            print('\033[95m' + 'INFO: Statistics are off, scores will not be plotted'+'\033[0m')

        env = self.env
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = self.agent

        scores, episodes = [], []

        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            lives = 3
            while not done:
                dead = False
                while not dead:
                    if agent.render:
                        env.render()

                    # get action for the current state and go one step in environment
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    # save the sample <s, a, r, s'> to the replay memory
                    agent.append_sample(state, action, reward, next_state, done)
                    # every time step do the training
                    agent.train_model()

                    state = next_state
                    score += reward
                    dead = info['ale.lives']<lives
                    lives = info['ale.lives']
                    # When Pacman dies gives penalty of -100
                    reward = reward if not dead else -100

                if done:
                    #print('\033[95m' + 'INFO: Done' + '\033[0m')
                    scores.append(score)
                    episodes.append(e)

                    if statistics:
                        pylab.plot(episodes, scores, 'b')
                        pylab.savefig(TRAINING_PATH + "pacman.png")

                    print('\033[92m' + 'INFO: Episode', e, "  Score", score, "  Memory Length", len(agent.memory), "  Epsilon", agent.epsilon, '\033[0m')

            # Save the model every 50 episodes if we are not in testing phase
            if (e % 50 == 0) & (mode.lower() != 'test'):
                agent.model.save_weights(TRAINING_PATH+"pacman.h5")
                print('\033[95m' +'INFO: Episode has ended, saving the network into the',TRAINING_PATH+'pacman.h5 file.' + '\033[0m')
            else:
                print('\033[95m' +'INFO: Episode has ended.' + '\033[0m')

        print('\033[95m' +'INFO: All episodes were run, exiting.' + '\033[0m')
