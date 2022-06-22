import argparse
import os
import time
import numpy as np
from distutils.util import strtobool
from mpi_utils import MPI_Tool
from stable_baselines3.common.evaluation import evaluate_policy
from utils.rl_tools import env_create_sb, env_create, eval_agent
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments

# from pbt_toy import pbt_engine
from mpi4py import MPI
from stable_baselines3 import PPO as PPO_SB
from ppo import PPO 
mpi_tool = MPI_Tool()
from tensorboardX import SummaryWriter

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HumanoidBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=141,
        help="seed of the experiment")
    parser.add_argument("--num-agents", type=int, default=20,
        help="number of agents")
    parser.add_argument("--num-envs", type=int, default=16,
        help="number of envs per agent")
    parser.add_argument("--total-generations", type=int, default=20,
        help="total generations of the experiments")
    parser.add_argument("--agent-training-steps", type=int, default=10000,
        help="total generations of the experiments")
    
    parser.add_argument("--learning-rate-range", type=tuple, default=(1e-4, 2e-3),
        help="the range of leanring rates among different agents")
    parser.add_argument("--gamma-range", type=tuple, default=(0.8, 0.99),
        help="the range of discount factors among different agents")
    parser.add_argument("--use-sb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be use stable-baselines")
    args = parser.parse_args()

    return args

class rl_agent():
    def __init__(self, idx, env_name, learning_rate, gamma, num_envs=1, use_sb=False, log_dir = "./tmp/gym/", seed=141) -> None:
        self.idx = idx
        self.seed = seed
        self.use_sb = use_sb
        self.score = 0 # For now just use reward per episode 
        self.length = 0 # For now just use length per episode 
        if env_name[0:8] == "MiniGrid":
            #self.env = env_create(env_name, idx)
            #self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
            #self.model =  PPO(envs=env_name, device='cpu', num_envs=8, verbose=0)
            if self.use_sb:
                #self.env = env_create(env_name, idx)
                self.env = make_vec_env(env_name, n_envs=num_envs)
                self.model = PPO_SB("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
                print("=="*10+"Stable-Baselines3 Agents"+"=="*10)
            else:
                self.model = PPO(envs=env_name, device='cpu', num_envs=num_envs, verbose=0)
                print("=="*10+"EVO Agents"+"=="*10)
        elif env_name[0:5] == "nasim":
            #self.env = env_create(env_name, idx)
            #self.env = env_create_sb(env_id=env_name, n_envs=8)
            #self.model =  PPO_SB("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
            if self.use_sb:
                #self.env = env_create(env_name, idx)
                self.env = make_vec_env(env_name, n_envs=num_envs)
                self.model =  PPO_SB("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
            else:
                #self.env = make_vec_env(env_name, n_envs=num_envs)
                self.model = PPO(envs=env_name, device='cpu', num_envs=num_envs, verbose=0, create_eval_env=True)
        elif env_name[0:6] == "dm2gym":
            self.env = env_create(env_name, idx)
            self.model = PPO("MultiInputPolicy", env=self.env, verbose=0, create_eval_env=True)
        elif env_name[0:3] == "Ant":
            self.env = env_create(env_name, idx)
            self.model = PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=True)
        else:
            self.model =  PPO("MlpPolicy", env=env_name, verbose=0, create_eval_env=True)
        self.model.gamma = gamma
        self.model.learning_rate = learning_rate
        self.log_dir = os.path.join(log_dir, str(idx))

    def step(self, traing_step=2000, callback=None, vanilla=False, rmsprop=False, Adam=False):
        """one episode of RL"""
        if self.use_sb:
            self.model.learn(total_timesteps=traing_step)
        else:
            self.model.train(total_timesteps=traing_step)
        

    def exploit(self, best_params):

        self.model.set_parameters(best_params) 
        

    def explore(self):
        """
        perturb hyperparaters with noise from a normal distribution
        """
        
        # LR 0.95 decay
        self.model.learning_rate=self.model.learning_rate*np.random.triangular(0.9, 0.95, 1.2)

        if self.model.gamma*np.random.uniform(0.9, 1.1)>=0.99:
            self.model.gamma = 0.99
        elif self.model.gamma*np.random.uniform(0.9, 1.1)<=0.8:
            self.model.gamma = 0.8
        else:
            self.model.gamma = self.model.gamma*np.random.uniform(0.9, 1.1)


    def eval(self, vanilla=True, return_episode_rewards=False):

        # Evaluate the agent

        if vanilla:
            if return_episode_rewards == True:
                if self.use_sb == True:
                    eps_reward, eps_length = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10, return_episode_rewards=True, deterministic=False)
                    eps_reward = np.mean(eps_reward)
                    eps_length = np.mean(eps_length)
                else:
                    eps_reward, eps_length = self.model.eval(num_eval_episodes=5)
                
                self.length = eps_length
            else:
                mean_reward, std_reward = self.model.eval(num_eval_episodes=5)
        else:
            #mean_reward = eval_agent(self.model, self.model.get_env())
            NotImplementedError

        #self.score =  eps_reward
        self.score =  eps_reward
        

    def update(self):
        """
        Just update the
        """

def workers_init(args):
    workers = []
    for idx in range(args.num_agents):
        # get learning rate, uniformly sampled on log scale
        _l_lb = np.log10(args.learning_rate_range[0])
        _l_ub = np.log10(args.learning_rate_range[1])
        if _l_ub >= _l_lb:       
            _lr = 10 ** np.random.uniform(low=_l_lb, high=_l_ub)
        else:
            raise Exception('Error in Learning Rate Range: Low bound shoud less that the Upper bound')
        
        # get discount factor, uniformly sampled 
        _g_lb = np.log10(args.gamma_range[0])
        _g_ub = np.log10(args.gamma_range[1])
        if _g_ub >= _g_lb:       
            _g = np.random.uniform(low=_g_lb, high=_g_ub)
        else:
            raise Exception('Error in Gamma Range: Low bound shoud less that the Upper bound')
        
        workers.append(rl_agent(idx=idx, env_name=args.env_id, learning_rate=_lr, gamma=_g, use_sb=args.use_sb, num_envs=args.num_envs)) 
    return workers

class base_population(object):
    def __init__(self):
        self.agents_pool = []

    def create(self, agent_list):
        self.agents_pool = agent_list

    def get_scores(self):
        return [worker.score for worker in self.agents_pool]
        # return score

    def get_best_agent(self):
        return self.get_scores().index(max(self.get_scores()))

    def get_best_score(self):
        # return max(self.get_scores())
        _best_id = self.get_best_agent()
        return self.agents_pool[_best_id].score
    
    def get_best_results(self):
        # return max(self.get_scores())
        _best_id = self.get_best_agent()
        return [self.agents_pool[_best_id].score, self.agents_pool[_best_id].length] 

    def get_best_agent_params(self):
        _best_id = self.get_best_agent()
        _best_agent = self.agents_pool[_best_id]
        params = _best_agent.model.get_parameters()

        return params

    @property
    def size(self):
        return int(len(self.agents_pool))


class base_engine(object):
    def __init__(self, total_population_size, tb_logger=None):
        self.total_population_size = total_population_size
        self.best_score_population = 0
        if mpi_tool.is_master & (tb_logger is not None):
            self.tb_writer = tb_logger
            #print(self.tb_writer)
        else:
            self.tb_writer = None

    def create_local(self, pbt_population):
        if pbt_population.size == 0:
            self.population = []
            self.best_params_population = []
        else:    
            self.population = pbt_population
            self.best_params_population = self.population.get_best_agent_params()
        

    def run(self, steps=3, exploit=False, explore=False, agent_training_steps=1000, return_episode_rewards=True):
        if not mpi_tool.is_master:
            print("Agents number: {} at rank {} on node {}".format(self.population.size, mpi_tool.rank, str(mpi_tool.node)))
        
        for i in range(steps):
            since = time.time()
            if mpi_tool.is_master:
                # Master is the centre controll, with no RL agent
                top=round(self.total_population_size*0.25)
                bottom=round(self.total_population_size*0.25)
                exchanged_vector = np.arange(self.total_population_size)
                #print(exchanged_vector)
            else:
                exchanged_vector = np.arange(self.total_population_size) 
                #print(exchanged_vector)

         
            for worker in self.population.agents_pool:
                worker.step(traing_step=agent_training_steps, vanilla=True)  # one step of GD
                worker.eval(return_episode_rewards=return_episode_rewards)
                
            # Update best score to the whole population
            if return_episode_rewards:
                best_results_to_sent = self.population.get_best_results()
            else:
                best_score_to_sent = self.population.get_best_score()
                
            best_params_to_sent = self.population.get_best_agent_params()
        
            if return_episode_rewards:
                #print(best_results_to_sent)
                best_score_to_sent, best_length_to_sent = best_results_to_sent[0], best_results_to_sent[1]
                best_scores = mpi_tool.gather(best_score_to_sent, root=0)
                best_length = mpi_tool.gather(best_length_to_sent, root=0)
            else:
                best_scores = mpi_tool.gather(best_score_to_sent, root=0)
            #print((best_scores, mpi_tool.rank))
            #mpi_tool.barrier()

            if i % 1 == 0 and i!=0:
                if mpi_tool.is_master:
                    """
                    scores: np.array([15, 10, 2, 8])
                    score_poistion: np.argsort(x) == array([2, 3, 1, 0])
                    exchanged_vector: [0,1,2,3]-->[0,1,0,3]
                    """
                    if return_episode_rewards:
                            #print(best_results.shape)
                            #best_scores, best_length = best_results
                        if best_scores is not None:
                            score_poistion = np.argsort(best_scores) 
                    else:
                        if best_scores is not None:
                            score_poistion = np.argsort(best_scores)
                    

                    if best_scores is not None:
                        for low_idx in score_poistion[:bottom]: 
                            exchanged_vector[score_poistion[low_idx]] = np.random.choice(score_poistion[-top:])

                    self.best_score_population = best_scores[score_poistion[-1]]
                    self.best_episode_length_population = best_length[score_poistion[-1]]
                    self.best_rank = score_poistion[-1]

                exchanged_vector = mpi_tool.bcast(exchanged_vector, root=0)

            
            #print((exchanged_vector, mpi_tool.rank))
            mpi_tool.barrier()
            #data = mpi_tool.rank
            for rec_idx in range(self.total_population_size):
                if rec_idx != exchanged_vector[rec_idx]:
                    #print(rec_idx)
                    #print(exchanged_vector[rec_idx])
                    #print(best_params_to_sent)
                    if mpi_tool.rank == exchanged_vector[rec_idx]:
                        MPI.COMM_WORLD.send(best_params_to_sent, dest=rec_idx, tag=rec_idx)
                    elif mpi_tool.rank == rec_idx:
                        best_params_to_sent=MPI.COMM_WORLD.recv(source=exchanged_vector[rec_idx], tag=rec_idx)

            #print(data, mpi_tool.rank)
            mpi_tool.barrier()
            if i % 1 == 0 and i!=0:
                for worker in self.population.agents_pool:
                    if explore and exploit:
                        #if worker.score <= rec_best_score:
                        
                        worker.exploit(best_params= best_params_to_sent)
                        worker.explore()
                    else:
                        pass
            
            
            mpi_tool.barrier()
            if mpi_tool.is_master:
                #self.best_score_population = rec_best_score
                # if return_episode_rewards:
                #     self.best_length_population = rec_best_length
                # self.best_params_population = best_params_population
                print("One Generation Time: {}".format(time.time()-since))
                if (i+1) % 1 == 0 and i!=0:
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('Score/PBT_Results', self.best_score_population, i)
                        self.tb_writer.add_scalar('Score/Training_Time_Per_Generation', time.time()-since, i)
                    if return_episode_rewards:
                        if self.tb_writer is not None:
                            self.tb_writer.add_scalar('Length/PBT_Results', self.best_episode_length_population, i)
                        #time.sleep(2.4)
                        print("At itre {} the Best Pop Score is {} Best Length is {} on rank {}".format(i, self.best_score_population, self.best_episode_length_population, self.best_rank ))
                    else:
                        print("At itre {} the Best Pop Score is {} on rank {}".format(i, self.best_score_population, self.best_rank))
    

def main():

    args = parse_args()
    run_name = f"{args.use_sb}_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    workers = workers_init(args)
    #writer = args.track
    
    if args.track:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None





    num_generations = args.total_generations
    agent_training_steps = args.agent_training_steps

 
    local_size, local_agent_inds = mpi_tool.split_size(len(workers))
    
    if local_size > 1:
        raise Exception('Updates! Each rank should only one single agent')
    else:
        print("Agent Number of {} at rank {}".format(local_agent_inds, mpi_tool.rank))

    # Initializing a local population
    print("{} at rank {}".format(local_agent_inds, mpi_tool.rank))

    pbt_population = base_population()
    pbt_population.create(agent_list=[workers[i] for i in local_agent_inds])

    # Initializing a local engin
    pbt_engine = base_engine(total_population_size=args.num_agents, tb_logger=writer)
    pbt_engine.create_local(pbt_population=pbt_population)

    run1 = pbt_engine.run(steps=num_generations,exploit=True, explore=True,agent_training_steps=agent_training_steps)
    if mpi_tool.is_master:
        if writer:
            pbt_engine.tb_writer.close()

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    if mpi_tool.is_master:
        print("Total Run Time: {}".format(time_elapsed))
