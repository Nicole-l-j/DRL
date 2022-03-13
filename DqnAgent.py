import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Vanilla_DQN.model import ConvDqn, DQN

import sys
sys.path.append('/home/ubuntu/data/lj/A_DRL_EXERCISE/Vanilla_DQN')


class DqnAgent(object):
    def __init__(self, opts):
        # ---------Define 2 networks (target and training)----------#
        self.opts = opts
        self.use_conv = self.opts.use_conv
        self.N_STATES = self.opts.n_states
        self.N_ACTIONS = self.opts.n_actions
        self.MEMORY_CAPACITY = self.opts.memory_capacity
        self.LR = self.opts.learning_rate
        self.EPSILON = self.opts.epsilon
        self.GAMMA = self.opts.gamma
        self.BATCH_SIZE = self.opts.batch_size
        if self.opts.use_conv:
            self.NET = ConvDqn(self.N_STATES, self.N_ACTIONS)
        else:
            self.NET = DQN(self.N_STATES, self.N_ACTIONS)
        self.eval_net, self.target_net = self.NET, self.NET
        # Define counter, memory size and loss function
        self.learn_step_counter = 0 # counter the steps of learning process
        self.memory_counter = 0 # counter used for experience replay buffer

        # --------Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_. the total is N_STATES * 2 + 2-------
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))

        # -----Define the optimizer ------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)

        # ----Define the loss function ------#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state
        # input only one sample
        if np.random.uniform() < self.EPSILON: # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            action = torch.max(actions_value, 1)[1].data.numpy()

            action = action[0] if self.opts.ENV_A_SHAPE == 0 else action.reshape(self.opts.ENV_A_SHAPE)
        else:
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.opts.ENV_A_SHAPE == 0 else action.reshape(self.opts.ENV_A_SHAPE)
        return action

    def store_transistion(self, s, a, r, s_):
        # This function acts as experience replay buffer
        transition = np.hstack((s, [a, r], s_)) # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # Update the target network every fixed steps
        if self.learn_step_counter % self.opts.TARGET_NETWORK_REPLACE_FREA == 0:
             # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Define the index of Sampled batch from buffer
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE) # Randomly select
        # extract experiences of batch size from the buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or metrices s, a, r, s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.N_STATES]))
        # Convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.N_STATES:]))

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of the next state
        q_next = self.target_net(b_s_).detach() # detach from computational graph,
        # select the maximun q value
        q_next.max(1)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1) # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step
