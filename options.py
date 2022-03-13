import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser(description='Vanilla DQN')

    parser.add_argument("--n_states", type=int, help='the shape of states of the game in gym')
    parser.add_argument("--n_actions", type=int, help='the number of actions which can be taken by the agent')
    parser.add_argument("--epsilon", type=float, default=0.9, help='used for the epsilon strategy')
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=3e-4, help='The learning rate for updating the parameters')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_episode", type=int, default=400)
    parser.add_argument("--max_step", type=int, default=200)
    parser.add_argument("--memory_capacity", type=int, default=2000)
    parser.add_argument("--TARGET_NETWORK_REPLACE_FREA", type=int, default=100)

    parser.add_argument("--use_conv", default=False)

    opts = parser.parse_args(args)

    return opts
