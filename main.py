
from envs import make_vec_envs
import argparse
import os
from distutils.util import strtobool
import torch
from ppo import PPO
import time

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=142,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nasim:HugeGen-v0",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=5,
        help="the number of parallel game environments")
    parser.add_argument("--no-obs-norm",type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="normali")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = make_vec_envs(args.env_id, args.seed, args.num_envs, args.gamma)

    model = PPO(envs=args.env_id, device=device, num_envs=args.num_envs, verbose=1)
    #model.train(15000)
    #model.eval(num_eval_episodes=2)
    #params = model.get_parameters()
    #model.set_parameters(params)
    model.train(50000)
    model.eval(num_eval_episodes=10)

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))