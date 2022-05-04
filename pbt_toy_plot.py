from re import S
import numpy as np
import time
from numpy.lib.utils import source
from utils.mpi_utils import MPI_Tool

from component import Agent, Population
# from pbt_toy import pbt_engine
from mpi4py import MPI
import matplotlib.pyplot as plt

mpi_tool = MPI_Tool()

class base_agent():
    def __init__(self, idx, obj_func, h, theta) -> None:
        self.idx = idx
        self.obj_func = obj_func
        #self.surrogate_obj_func = lambda theta, h: 1.2 - np.sum(h*theta**2)

        self.h = h
        self.theta = theta
        self.rms = 0

        self.score = 0

        self.theta_history = []
        self.loss_history = []

    def step(self, vanilla=False, rmsprop=False, Adam=False):
        """one step of GD"""
        decay_rate = 0.9
        alpha = 0.01
        eps = 1e-5

        d_surrogate_obj = -2.0 * self.h * self.theta

        if vanilla:
           
            self.theta += d_surrogate_obj * alpha  # ascent to maximize function
          
        else:
            self.rms = decay_rate * self.rms + \
                (1-decay_rate) * d_surrogate_obj**2
            self.theta += alpha * d_surrogate_obj / (np.sqrt(self.rms) + eps)

    def exploit(self, best_params):
        """
        copy weights, hyperparams from the member in the population with
        the highest performance

        pop_score is a Dict, thus
        https://stackoverflow.com/questions/61918145/how-works-python-key-operator-itemgetter1
        """
        # best_worker_idx = max(self.pop_score.items(),
        #                       key=operator.itemgetter(1))[0]
        # if best_worker_idx != self.idx:
        #print(best_params)
        best_worker_theta, best_worker_h = best_params
        # print(best_worker_theta)
        self.theta = np.copy(best_worker_theta)
        #self.h = np.copy(best_worker_h)

        # return False
    
    
    def explore(self):
        """
        perturb hyperparaters with noise from a normal distribution
        """
       
        eps = np.random.uniform(-0.000000, 0.000005, 2) 
        #print(eps)
        self.h = self.h+eps

    def eval(self):
        self.score = self.obj_func(self.theta)
        # return self.score

    def update(self):
        """
        Just update the loss hist
        """
        self.theta_history.append(np.copy(self.theta))
        self.loss_history.append(self.score)


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

    def get_best_agent_params(self):
        _best_id = self.get_best_agent()
        _best_agent = self.agents_pool[_best_id]
        params = (np.copy(_best_agent.theta), np.copy(_best_agent.h))
        # print(params)
        return params

    @property
    def size(self):
        return int(len(self.agents_pool))


class base_engine(object):
    def __init__(self):
        self.best_score_population = 0
        
    def create_local(self, pbt_population):

        self.population = pbt_population
        self.best_params_population = self.population.get_best_agent_params()

    def run(self, steps=3, exploit=False, explore=False):
        print("Agents number: {} at rank {} on node {}".format(
            self.population.size, mpi_tool.rank, str(mpi_tool.node)))
        for i in range(steps):

            if mpi_tool.is_master:
                best_score_at_each_step = self.best_score_population
                best_params_at_each_step = self.best_params_population
            else:
                best_score_at_each_step = None
                best_params_at_each_step = None
            best_score_at_each_step = mpi_tool.bcast(best_score_at_each_step, root=0)
            best_params_at_each_step = mpi_tool.bcast(best_params_at_each_step, root=0)

            for worker in self.population.agents_pool:
                worker.step(vanilla=False)  # one step of GD
                worker.eval()
            # Update best score to the whole population
            best_score_to_sent = self.population.get_best_score()
            best_params_to_sent = self.population.get_best_agent_params()

            rec_best_score, best_score_rank = MPI.COMM_WORLD.allreduce((best_score_to_sent, mpi_tool.rank), op=MPI.MAXLOC)
            #print("Best Score {} on Rank {} at Step{}".format(rec_best_score, best_score_rank, i))
            if mpi_tool.rank == best_score_rank:
                best_params_population = best_params_to_sent
            else:
                best_params_population = None
            
            best_params_population = mpi_tool.bcast(best_params_to_sent, root=best_score_rank)

            if i % 20 == 0:
                for worker in self.population.agents_pool:
                    if explore and exploit:
                        #print("My Score {} ar rank {} Highest Score Now: {} at rank {}".format(worker.score,mpi_tool.rank,rec_best_score, best_score_rank))
                        if worker.score < rec_best_score:
                            worker.exploit(best_params=best_params_population)
                            worker.explore()
                    elif explore and not exploit:
                        worker.explore()
                
                    elif not explore and exploit:
                        if worker.score < rec_best_score:
                            worker.exploit(best_params=best_params_population)
                    
                    else:
                        pass
            
            for worker in self.population.agents_pool:
                worker.update()

            if mpi_tool.is_master:
                self.best_score_population = rec_best_score
                self.best_params_population = best_params_population

                if (i+1) % 10== 0:
                    print("At itre {} the Best Pop Score is {}".format(
                       i, self.best_score_population))
        return self.population   

def plot_loss(run, i, steps, title, color):
    
    plt.subplot(2,4,i)
    plt.plot(run.agents_pool[0].loss_history, color=color, lw=0.7)
    plt.plot(run.agents_pool[1].loss_history, color='r', lw=0.7)
    plt.axhline(y=1.2, linestyle='dotted', color='k')
    axes = plt.gca()
    axes.set_xlim([0,steps])
    axes.set_ylim([0.0, 1.21])
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Q')

    
def plot_theta(run, i, steps, title, color):
    x_b = [_[0] for _ in run.agents_pool[0].theta_history]
    y_b = [_[1] for _ in run.agents_pool[0].theta_history]
    
    x_r = [_[0] for _ in run.agents_pool[1].theta_history]
    y_r = [_[1] for _ in run.agents_pool[1].theta_history]
    
    plt.subplot(2,4,i)
    plt.scatter(x_b, y_b, color=color, s=2)
    plt.scatter(x_r, y_r, color='r', s=2)
    
    plt.title(title)
    plt.xlabel('theta0')
    plt.ylabel('theta1')

def run_exp(obj, steps, exploit=True, explore=True):
    

    wokers = [base_agent(idx=0, obj_func=obj, h=np.array([1., 0]), theta=np.array([.9, .9])),
              base_agent(idx=1, obj_func=obj, h=np.array([0, .1]), theta=np.array([.9, .9]))]

    local_size, local_agent_inds = mpi_tool.split_size(len(wokers))
    print("Agent Number of {} at rank {}".format(
        local_agent_inds, mpi_tool.rank))

    # Initializing a local population
    pbt_population = base_population()
    pbt_population.create(agent_list=[wokers[i] for i in local_agent_inds])

    # Initializing a local engin

    pbt_engine = base_engine()
    pbt_engine.create_local(pbt_population=pbt_population)

    run = pbt_engine.run(steps=steps, exploit=exploit, explore=explore)

    return run

def main():
    color_vec = ['b', 'r']
    steps = 200
    def obj(theta): return 1.2 - np.sum(theta**2) 
    
    run1 = run_exp(obj=obj, steps=steps, exploit=True, explore=True)
    run2 = run_exp(obj=obj, steps=steps, exploit=False, explore=True)
    run3 = run_exp(obj=obj, steps=steps, exploit=True, explore=False)
    run4 = run_exp(obj=obj, steps=steps, exploit=False, explore=False)

    # plot_loss(run1, 3, steps=steps, title='PBT', color = color_vec[mpi_tool.rank])
    # plot_theta(run1, 4, steps=steps, title='PBT', color = color_vec[mpi_tool.rank])


    plot_loss(run1, 3, steps=steps, title='PBT', color = color_vec[mpi_tool.rank])
    plot_loss(run2, 4, steps=steps, title='Explore only', color = color_vec[mpi_tool.rank])
    plot_loss(run3, 7, steps=steps, title='Exploit only', color = color_vec[mpi_tool.rank])
    plot_loss(run4, 8, steps=steps, title='Grid Search', color = color_vec[mpi_tool.rank])
    
    plot_theta(run1, 1, steps=steps, title='PBT', color = color_vec[mpi_tool.rank])
    plot_theta(run2, 2, steps=steps, title='Explore only', color = color_vec[mpi_tool.rank])
    plot_theta(run3, 5, steps=steps, title='Exploit only', color = color_vec[mpi_tool.rank])
    plot_theta(run4, 6, steps=steps, title='Grid Search', color = color_vec[mpi_tool.rank])
    
if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    if mpi_tool.is_master:
        print("Total Run Time: {}".format(time_elapsed))
    plt.show()
