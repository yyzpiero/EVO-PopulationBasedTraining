import argparse
import time
import os
import numpy as np

from stable_baselines3 import PPO
from utils.mpi_utils import MPI_Tool
from utils.rl_tools import env_create_sb, env_create, eval_agent
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter

mpi_tool = MPI_Tool()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--tb-writer", type=bool, default=False,
        help="if toggled, Tensorboard summary writer is enabled")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="AntBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=141,
        help="seed of the experiment")
    parser.add_argument("--num-agents", type=int, default=20,
        help="number of agents")
    parser.add_argument("--total-generations", type=int, default=250,
        help="total generations of the experiments")
    parser.add_argument("--agent-training-steps", type=int, default=2000,
        help="total generations of the experiments")
    
    parser.add_argument("--learning-rate-range", type=tuple, default=(1e-4, 1e-3),
        help="the range of leanring rates among different agents")
    parser.add_argument("--gamma-range", type=tuple, default=(0.8, 0.99),
        help="the range of discount factors among different agents")
    args = parser.parse_args()

    return args

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

        workers.append(rl_agent(idx=idx, env_name=args.env_id, learning_rate=_lr, gamma=_g))
    return workers

class rl_agent(object):

    def __init__(self, idx, env_name,learning_rate, gamma, log_dir = "./tmp/zjy/", seed=141):
        self.idx = idx
        self.seed = seed + 100*mpi_tool.rank
        self.score = 0 # For now just use reward per episode 
        self.length = 0 # For now just use length per episode 
        
        if env_name[0:8] == "MiniGrid":
            self.env = env_create(env_name, idx, seed=self.seed)
            self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False, seed=self.seed)
        elif env_name[0:5] == "nasim":
            self.env = env_create(env_name, idx, seed=self.seed)
            self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False, seed=self.seed)
        elif env_name[0:6] == "dm2gym":
            self.env = env_create(env_name, idx, seed=self.seed)
            self.model = PPO("MultiInputPolicy", env=self.env, verbose=0, create_eval_env=True, seed=self.seed)
        elif env_name[-12:-6] == "Bullet":
            self.env = env_create(env_name, idx, seed=self.seed)
            self.model = PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=True, seed=self.seed)
        else:
            self.model =  PPO("MlpPolicy", env=env_name, verbose=0, create_eval_env=True)
        
        self.log_dir = os.path.join(log_dir, str(idx))
        self.model.gamma = gamma
        self.model.learning_rate = learning_rate
        self.params = self.model.get_parameters()

    def step(self, traing_step=2000, callback=None, vanilla=False, rmsprop=False, Adam=False):
        """one episode of RL"""
        self.model.learn(total_timesteps=traing_step)#, callback=callback)
      
    def exploit(self, best_params):
        """
        copy weights, hyperparams from the member in the population with
        the highest performance

        pop_score is a Dict, thus
        https://stackoverflow.com/questions/61918145/how-works-python-key-operator-itemgetter1
        """
        self.model.set_parameters(best_params) 

    def explore(self):
        """
        perturb hyperparaters with noise from a normal distribution
        """
        self.model.learning_rate=self.model.learning_rate*np.random.triangular(0.9, 0.95, 1.2)

        if self.model.gamma*np.random.uniform(0.9, 1.1)>=0.99:
            self.model.gamma = 0.99
        elif self.model.gamma*np.random.uniform(0.9, 1.1)<=0.8:
            self.model.gamma = 0.8
        else:
            self.model.gamma = self.model.gamma*np.random.uniform(0.9, 1.1) 

    def eval(self, vanilla=True, return_episode_rewards=False):

        # Evaluate the agent
        # NOTE: If you use wrappers with your environment that modify rewards,
        #       this will be reflected here. To evaluate with original rewards,
        #       wrap environment in a "Monitor" wrapper before other wrappers.
        if vanilla:
            if return_episode_rewards == True:
                eps_reward, eps_length = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10, return_episode_rewards=True)
                mean_reward = np.mean(eps_reward)
                mean_length = np.mean(eps_length)
                self.length = mean_length
            else:
                mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
        else:
            mean_reward = eval_agent(self.model, self.model.get_env())
        self.score =  mean_reward
        if mpi_tool.is_master:
            print("mean reward:",mean_reward)

class dummy_worker(object):
    def __init__(self,worker,rank): 
        self.idx=worker.idx   
        self.rank=rank
        self.score=worker.score
        self.length=worker.length
        self.results=[worker.score, worker.length]
        self.params=worker.model.get_parameters()     
        self.gamma=worker.model.gamma
        self.lr=worker.model.learning_rate

class base_population(object):
    def __init__(self):
        self.agents_pool = []
        # self.rank=0
        # self.idx_list=[]
    def create(self, agent_list,rank,idx_list):
        self.agents_pool = agent_list
        self.rank=rank
        self.idx_list=idx_list
        
    def get_scores(self):
        return [worker.score for worker in self.agents_pool]

    def get_best_agent(self):
        return self.get_scores().index(max(self.get_scores()))
    
    def get_best_score(self):
        _best_id = self.get_best_agent()
        return self.agents_pool[_best_id].score

    def get_best_results(self):
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
    def __init__(self, tb_logger=False, length_first=False):
        
        self.best_score_population = 0
        self.best_episode_length_population = 0
        self.length_first = length_first

        if mpi_tool.is_master & (tb_logger):
            self.tb_writer = SummaryWriter()
        else:
            self.tb_writer = False

    def create_local(self, pbt_population):

        self.population = pbt_population
        self.best_params_population = self.population.get_best_agent_params()

    def run(self, steps=3, exploit=False, explore=False, agent_training_steps=1000, return_episode_rewards=True):

        print("Agents number: {} at rank {} on node {}".format(
            self.population.size, mpi_tool.rank, str(mpi_tool.node)))

        for i in range(steps):

            for worker in self.population.agents_pool:
                worker.step(traing_step=agent_training_steps, vanilla=True)  # one step of GD
                worker.eval(return_episode_rewards=return_episode_rewards)

            if len(self.population.agents_pool)==1:
                worker = self.population.agents_pool[0]

            if mpi_tool.is_master:
                rl_list=[]
                flag_list = [False for i in range(mpi_tool.size)]
            else:
                rl_list = None
                flag_list = None
                top_fitness_params = None

            rl_worker=dummy_worker(worker, mpi_tool.rank)
            rl_list=mpi_tool.gather(rl_worker, root=0)

            if mpi_tool.is_master:
                top_num=round(len(rl_list)*0.3)
                bottom_num=round(len(rl_list)*0.3)

                if return_episode_rewards:
                    top_length_idx = np.argsort([w.length for w in rl_list])[-top_num:]
                top_score_idx = np.argsort([w.score for w in rl_list])[-top_num:]
                
                if self.length_first:
                    top_fitness_params = [rl_list[j].params for j in top_length_idx]
                else:
                    top_fitness_params = [rl_list[j].params for j in top_score_idx]

                bottom_score_idx = [rl_list[j].idx for j in np.argsort([w.score for w in rl_list])[:bottom_num]]
                for j in bottom_score_idx:
                    flag_list[j] = True

            top_fitness_params = mpi_tool.bcast(top_fitness_params,root=0)
            bottom_flag = mpi_tool.scatter(flag_list, root=0)              

            if i % 1 == 0 and i!=0:
                for worker in self.population.agents_pool:
                    if explore and exploit:
                        if bottom_flag:
                            best_params_to_sent=np.random.choice(top_fitness_params)
                            worker.exploit(best_params=best_params_to_sent)
                        else:
                            pass    
                        worker.explore()
                    else:
                        pass

            if mpi_tool.is_master:
                self.best_score_population = np.max([worker.score for worker in rl_list])
                self.best_episode_length_population = np.min([worker.length for worker in rl_list])
                if (i+1) % 1 == 0 and i!=0:
                    if return_episode_rewards:
                        print("At itre {} the Best Pop Score is {} Best Length is {}".format(i, self.best_score_population, self.best_episode_length_population))
                        if self.tb_writer:
                            self.tb_writer.add_scalar('Score/PBT_Results', self.best_score_population, i)
                            self.tb_writer.add_scalar('Length/PBT_Results', self.best_episode_length_population, i)
                    else:
                        print("At itre {} the Best Pop Score is {}".format(i, self.best_score_population))
                        if self.tb_writer:
                            self.tb_writer.add_scalar('Score/PBT_Results', self.best_score_population, i)
                        
                        

def main():
    args = parse_args()
    
    if args.env_id[0:5] == "nasim" or args.env_id[0:8] == "MiniGrid":
        length_first = True
    else:
        length_first = False
    
    writer = args.tb_writer
    num_generations = args.total_generations
    agent_training_steps = args.agent_training_steps
    
    workers = workers_init(args)
    local_size, local_agent_inds = mpi_tool.split_size(len(workers))
    print("Agent Number of {} at rank {}".format(local_agent_inds, mpi_tool.rank))

    # Initializing a local population
    pbt_population = base_population()
    pbt_population.create(agent_list=[workers[i] for i in local_agent_inds],rank=mpi_tool.rank,idx_list=[workers[i].idx for i in local_agent_inds])

    # Initializing a local engin
    pbt_engine = base_engine(tb_logger=writer, length_first=length_first)
    pbt_engine.create_local(pbt_population=pbt_population)

    run1 = pbt_engine.run(steps=num_generations,exploit=True, explore=True, agent_training_steps=agent_training_steps)
    if mpi_tool.is_master and writer:
        pbt_engine.tb_writer.close()

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    if mpi_tool.is_master:
        print("Total Run Time: {}".format(time_elapsed))

