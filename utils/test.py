import sys
import os
from mpi4py import MPI
import numpy as np
from mpi_tools import mpi_fork

def func(params_dict_list):
    proc_id = MPI.COMM_WORLD.Get_rank()
    if proc_id > len(params_dict_list)-1:
        print("proc_id:", proc_id)
        print("sys.exit()")
        sys.exit()
        print("sys.exit()")
    params_dict = params_dict_list[proc_id]
    print("proc_id:", proc_id)
    print("params_dict:", params_dict)
    print("-"*20)

def func_mpi(params_dict_list):
    proc_id = MPI.COMM_WORLD.Get_rank()
    if proc_id > len(params_dict_list)-1:
        print("proc_id:", proc_id)
        print("sys.exit()")
        sys.exit()
        print("sys.exit()")
    params_dict = params_dict_list[proc_id]
    

if __name__=='__main__':
    params_dict = {
        'lr': [2, 3, 4, 5, 6, 7],
        "batch": [10, 20, 30, 40, 50,],
        "epoch": [100, 200, 300, 400, 500, 600],
    }
    import itertools
    
    params_list = [list(value) for value in itertools.product(*params_dict.values())]        
    params_dict_list = [{key: cur_param.pop(0) for key, value in params_dict.items()} for cur_param in params_list]
    print(params_dict_list)
    print('-'*40)
    for i in range(0, len(params_dict_list), 3):
        chunk_params_dict_list = params_dict_list[i:i+3]        
        func(params_dict_list=chunk_params_dict_list)