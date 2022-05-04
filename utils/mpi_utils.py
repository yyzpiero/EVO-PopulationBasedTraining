#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A high-level utility class for parallelizing Genetic Algorithm by using MPI interfaces
in distributed MPI environment.
'''

import logging
import numpy as np
from itertools import chain
from functools import wraps

try:
    from mpi4py import MPI
    MPI_INSTALLED = True
except ImportError:
    MPI_INSTALLED = False


class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


class MPI_Tool(object):

    __metaclass__ = Singleton

    def __init__(self):
        ''' Wrapper class for higher level of MPI interfaces that will create a
        singleton for parallelization.
        '''
        logger_name = 'gaft.{}'.format(self.__class__.__name__)
        self._logger = logging.getLogger(logger_name)

    def gather(self, data, root=0):
        ''' Gatehr data to MPI processes
        :param data: Data to be gathered
        :type data: any Python object
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            bdata = mpi_comm.gather(data, root=root)
        else:
            bdata = data
        return bdata
    
    def scatter(self, data, root=0):
        ''' Scatter data to MPI processes
        :param data: Data to be gathered
        :type data: any Python object
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            bdata = mpi_comm.scatter(data, root=root)
        else:
            bdata = data
        return bdata

    def bcast(self, data, root=0):
        ''' Broadcast data to MPI processes
        :param data: Data to be broadcasted
        :type data: any Python object
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            bdata = mpi_comm.bcast(data, root=root)
        else:
            bdata = data
        return bdata

    def sendrecv(self, data, dest, source):
        ''' SendRecv data to MPI processes
        :param data: Data to be broadcasted
        :type data: any Python object
        :dest: Destnition Rank
        :source: Source Rank int
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            bdata = mpi_comm.sendrecv(data, dest, source)
        else:
            bdata = data
        return bdata

    def msg(m, string=''):
        if MPI_INSTALLED:
            print(('Message from %d: %s \t ' %
                  (MPI.COMM_WORLD.Get_rank(), string))+str(m))
        else:
            print(('Message: %s \t' % string)+str(m))

    def get_rank(self):
        ''' Get the rank of the calling process in the communicator
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            return mpi_comm.Get_rank()
        else:
            return 0

    # Global Methods

    def allreduce_helper(*args, **kwargs):
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

    def mpi_op(self, x, op):

        mpi_comm = MPI.COMM_WORLD
        x, scalar = ([x], True) if np.isscalar(x) else (x, False)
        x = np.asarray(x, dtype=np.float32)
        buff = np.zeros_like(x, dtype=np.float32)
        self.allreduce_helper(x, buff, op=op)
        return buff[0] if scalar else buff

    def mpi_sum(self, x):
        if MPI_INSTALLED:
            return self.mpi_op(x, MPI.SUM)
        else:
            return x

    def mpi_max(self, x):
        return self.mpi_op(x, MPI.MAX)

    def mpi_avg(self, x):
        """Average a scalar or vector over MPI processes."""
        return self.mpi_sum(x) / self.size()

    def mpi_statistics_scalar(self, x, with_min_and_max=False):
        """
        Get mean/std and optional min/max of scalar x across MPI processes.
        Args:
            x: An array containing samples of the scalar to produce statistics
                for.
            with_min_and_max (bool): If true, return min and max of x in 
                addition to mean and std.
        """
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = self.mpi_sum([np.sum(x), len(x)])
        mean = global_sum / global_n

        global_sum_sq = self.mpi_sum(np.sum((x - mean)**2))
        std = np.sqrt(global_sum_sq / global_n)  # compute global std

        if with_min_and_max:
            global_min = self.mpi_op(
                np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
            global_max = self.mpi_op(np.max(x) if len(
                x) > 0 else -np.inf, op=MPI.MAX)
            return mean, std, global_min, global_max
        return mean, std

    # Wrapper for common MPI interfaces.
    def barrier(self):
        ''' Block until all processes in the communicator have reached this routine
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            mpi_comm.barrier()

    @property
    def rank(self):
        ''' Get the rank of the calling process in the communicator
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            return mpi_comm.Get_rank()
        else:
            return 0
    @property
    def node(self):
        if MPI_INSTALLED:
            #mpi_comm = MPI.COMM_WORLD
            return MPI.Get_processor_name()
        else:
            return 0
        #node_name = MPI.Get_processor_name()
        
    @property
    def size(self):
        ''' Get the size of the group associated with a communicator
        '''
        if MPI_INSTALLED:
            mpi_comm = MPI.COMM_WORLD
            return mpi_comm.Get_size()
        else:
            return 1

    @property
    def is_master(self):
        ''' If current process is the master process
        '''
   
        return self.rank == 0
        

    # Utility methods.
    def split_seq(self, sequence):
        ''' Split the sequence according to rank and processor number.
        :param sequence: Data sequence to be splitted
        :type sequence: any Python object list
        :return: Sub data sequence for current process
        :rtype: any Python object list
        '''
        starts = [i for i in range(0, len(sequence), len(sequence)//self.size)]
        ends = starts[1:] + [len(sequence)]
        start, end = list(zip(starts, ends))[self.rank]

        return sequence[start: end]

    def split_size(self, size, ignore_master = False):
        ''' Split a size number(int) to sub-size number.
        :param size: The size number to be splitted.
        :type size: int
        :return: Sub-size for current process
        :rtype: int
        '''
        if ignore_master:
            if size < self.size-1:
                warn_msg = ('Splitting size({}) is smaller than process ' +
                            'number({}), more processor would be ' +
                            'superflous').format(size, self.size)
                self._logger.warning(warn_msg)
                splited_sizes = [1]*size + [0]*(self.size - 1 - size)
                indexs_list = [[i]
                            for i in range(size)] + [None]*(self.size-1 - size)
            elif size % (self.size-1) != 0:
                residual = size % (self.size-1)
                local_size = size // (self.size-1)
                splited_sizes = [local_size]*(self.size-1)
                indexs_list = [[*range((i) * local_size, (i+1) * local_size)]
                            for i in range(size)]
                for i in range(residual):
                    splited_sizes[i+1] += 1
                    indexs_list[i+1].extend([self.size+i])
            else:
                local_size = size // (self.size-1)
                splited_sizes = [local_size]*(self.size-1)
                indexs_list = [[*range((i) * local_size, (i+1) * local_size)]
                            for i in range(size)]
            return splited_sizes[self.rank], indexs_list[self.rank]

        else:
            if size < self.size:
                warn_msg = ('Splitting size({}) is smaller than process ' +
                            'number({}), more processor would be ' +
                            'superflous').format(size, self.size)
                self._logger.warning(warn_msg)
                splited_sizes = [1]*size + [0]*(self.size - size)
                indexs_list = [[i]
                            for i in range(size)] + [None]*(self.size - size)
            elif size % self.size != 0:
                residual = size % self.size
                local_size = size // self.size
                splited_sizes = [local_size]*self.size
                indexs_list = [[*range((i) * local_size, (i+1) * local_size)]
                            for i in range(size)]
                for i in range(residual):
                    splited_sizes[i] += 1
                    indexs_list[i].extend([self.size+i])
            else:
                local_size = size // self.size
                splited_sizes = [local_size]*self.size
                indexs_list = [[*range((i) * local_size, (i+1) * local_size)]
                            for i in range(size)]

            return splited_sizes[self.rank], indexs_list[self.rank]

    def gather_seq(self, seq):
        ''' Gather data in sub-process to root process.
        :param seq: Sub data sequence for current process
        :type seq: any Python object list
        :return: Merged data sequence from all processes in a communicator
        :rtype: any Python object list
        '''
        if self.size == 1:
            return seq

        mpi_comm = MPI.COMM_WORLD
        merged_seq = mpi_comm.allgather(seq)
        return list(chain(*merged_seq))  # iterate over all the sequence

    def merge_seq(self, seq, op="MAX"):
        ''' Merge data in sub-process to root process.
        :param seq: Sub data sequence for current process
        :type seq: any Python object list
        :return: Merged data sequence from all processes in a communicator
        :rtype: any Python object list
        '''
        if self.size == 1:
            return seq

        mpi_comm = MPI.COMM_WORLD
        if op == "SUM":
            merged_seq = mpi_comm.allreduce(seq, MPI.SUM)
        elif op == "MAX":
            merged_seq = mpi_comm.allreduce(seq, op=MPI.MAX)
        return merged_seq


def master_only(func):
    ''' Decorator to limit a function to be called only in master process in MPI env.
    '''
    @wraps(func)
    def _call_in_master_proc(*args, **kwargs):
        mpi = MPI_Tool()
        if mpi.is_master:
            return func(*args, **kwargs)

    return _call_in_master_proc
