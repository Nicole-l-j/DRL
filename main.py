import gym
import sys

from Vanilla_DQN.DqnAgent import DqnAgent
from Vanilla_DQN.options import get_options

sys.path.append('/home/ubuntu/data/lj/A_DRL_EXERCISE/Vanilla_DQN')
env = gym.make("CartPole-v0").unwrapped

opts = get_options()
opts.n_states = env.observation_space.shape[0]
opts.n_actions = env.action_space.n

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample()
opts.ENV_A_SHAPE = ENV_A_SHAPE

dqn = DqnAgent(opts)
for i_episode in range(opts.max_episode):
    s = env.reset()
    episode_reward = 0

    while True:

        env.render()
        # Take action based on the current state
        a = dqn.choose_action(s)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)

        # modify the reward based on the environment state
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store the transitions of states
        dqn.store_transistion(s, a, r, s_)

        episode_reward += r
        #  if the experience replay buffer is filled, DQN begins to learn or update
        # its parameters.
        if dqn.memory_counter > opts.memory_capacity:
            dqn.learn()
            if done:
                print('Ep:', i_episode, '|', 'Ep_r:', round(episode_reward, 2))

        if done:
            # if game is over, then skip the while loop.
            break
            # use next state to update the current state.
        s = s_
