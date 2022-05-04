# EVO: Population-Based Training (PBT) for Reinforcement Learning using MPI 
## Overview

Population Based Training is a novel approach to hyperparameter optimisation by jointly optimising a population of models and their hyperparameters to maximise performance. PBT takes its inspiration from genetic algorithms where each member of the population can exploit information from the remainder of the population.



<p>
    <img src="https://i.imgur.com/hvfgzyf.png" alt="PBT Illustration" style="zoom:50%;" />
</p>
<p>
    <em>Illustration of PBT training process (Liebig, Jan Frederik, Evaluating Population based Reinforcement Learning for Transfer Learning, 2021)</em>
</p>


To extend the population of agents to extrem scale using High Performce Computer, this repo, namely **EVO** provide a PBT implementation for RL using Message Passing Interface. 

## MPI (Message Passing Interface) and mpi4py

[Message passing interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) provides a powerful, efficient, and portable way to express parallel programs.  It is the dominant model used in [high-performance computing](https://en.wikipedia.org/wiki/High-performance_computing). 

[mpi4py](https://mpi4py.readthedocs.io/en/stable/) provides a Python interface that resembles the [message passing interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface), and hence allows Python programs to exploit multiple processors on multiple compute nodes. 

## Get Started
Prerequisites:

- Python 3.8
- Conda
- (Poetry)
- (Pytorch)[^1]

[^1]: Please use **cpu-only** version if possible, as most HPC clusters don't have GPUs

Clone the repo: 

```
git clone https://github.com/yyzpiero/evo.git
```

Create conda environment:
```
conda create -p ./venv
```
and use poetry to install all Python packages:

```
poetry install
```

Please use pip or poetry to install `mpi4py` :  
```
pip install mpi4py
``` 

or

```
poetry add mpi4py
```

As using Conda install may lead to some unknown issues.


### Basic Usage
Activate conda environment:
```
conda activate ./venv
```

Please use `mpiexec` or `mpirun` to run experiments:
```
mpiexec -n 4 python pbt_rl_wta.py --num-agents 4 --env-id CartPole-v1
```


### Example

#### Tensorboard support
EVO also supports experiment monitoring with Tensorboard. Example command line to run an experiment with Tensorboard monitoring:
```
mpiexe -n 4 python pbt_rl_truct_collective.py --num-agents 4 --env-id CartPole-v1 --tb-writer True
```
## Toy Model
The toy example was reproduced from Fig. 2 in the [PBT paper](https://arxiv.org/abs/1711.09846)

<p>
    <img src="https://i.imgur.com/bbJ12k5.png" alt="PBT Illustration" style="zoom:50%;" />
</p>

## Reinforcement Learning Agent

[PPO agent]() from [`stable-baselines 3`](https://github.com/DLR-RM/stable-baselines3) with default settings are used as reinforcement learning agent.

` self.model = PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=True)` 

However, it can also be replaced by any other reinforcement learning algorithms.

### Reference: 

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [PPO in Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Selection Mechanism

### "Winner-takes-all"

A simply selection mechanism, that for each generation, only the best-performed agent is kept, and its NN parameters are copied to all other agents. 
.py provides an implementation of such a mechanism using collective communications.

### Truncation selection

> It the default selection strategy in [PBT paper](https://arxiv.org/abs/1711.09846) for RL training, and is widely used other PBT-based methods.

All agents in the entire population are ranked by their episodic rewards. It the a agent is in the bottom $25\%$ of the entire population, another agent from the top $25\%$ is sampled and its NN parameters and hyperparameters are copied to the current agent.  **Different <u>MPI communication methods</u>[^note] are implementated.**

#### Implemented Variants

| Variants | Description                                                  |
| -------- | ------------------------------------------------------------ |
|   `pbt_rl_truct.py`       | implementation using point-2-point communications via `send` and `recv`. |
|     `pbt_rl_truct_collective.py`     | implementation using collective communications.              |

For small cluster with limited number of nodes, we suggest point-2-point method, which is more faster than collective method. However, for large HPC cluster, collective method is much more faster and robust.

[^note]: [This article](https://www.futurelearn.com/info/courses/python-in-hpc/0/steps/65143) briefly intorduces the difference between point-2-point communcations and collective communicatiosn in MPI.

## Benchmarks

We used continuous control `AntBulletEnv-v0` scenario in [PyBullet environments](https://pybullet.org/wordpress/) to test our implementations. 

Results of the experiments are presented on the Figure below:


<p>
    <img src="https://i.imgur.com/Mi5Giit.png" alt="Benchmark Results" style="zoom:50%;" />
</p>

<p>
    <em>Left Figure: Reward per generation using PBT | Right Figure: Reward per step using single SB3 agent</em>
</p>




**Some key observations:**

- By using PBT to train PPO agent can acheive better results than SAC (single agent)

  - Note: SAC should outperforms PPO (see [OpenRL](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Open-RL-Benchmark-0-6-0---Vmlldzo0MDcxOA)) in most *PyBullet* environments

- "Winner-takes-all" outperforms the Truncation Selection mechanism in this scenario.


## Acknowledgements
This repo is inspired by [graf](https://github.com/PytLab/gaft), [angusfung's population based training repo](https://github.com/angusfung/population-based-training). 
