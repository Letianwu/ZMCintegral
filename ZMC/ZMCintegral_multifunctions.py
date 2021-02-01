#!/usr/bin/env python
# coding: utf-8

# In[81]:


import math
import numpy as np
from numba import cuda
import numba as nb
import random
import os
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import ray
import itertools
import time


# In[82]:


@ray.remote(num_gpus=1)
def gpu_nums():
    # return all detected available gpus, int
    detected_gpu_id = ray.cluster_resources()['GPU']
    return detected_gpu_id


# In[83]:


@ray.remote(num_gpus=1)
def MCkernel(domain, my_func, num_points):
    
    """
    Params
    ======
    domain: domain of the current high-dim-function
    my_func: function that need to be integrated
    num_points: number of the sampling points
    
    Return
    ======
    result: integration result of the current function
    """
            
    @cuda.jit
    def integration_kernel(MCresult, domain, domain_range, rng_states, num_points):

        thread_id = cuda.grid(1)
        
        if thread_id < num_points:
            # local array to save random numbers
            x_tuple = cuda.local.array(shape=dim, dtype=nb.float64)

            for j_dim in range(dim):
                x_tuple[j_dim] = xoroshiro128p_uniform_float64(rng_states, thread_id) * domain_range[j_dim] + domain[j_dim][0]

            # accumulate the sampled results on global memory
            cuda.atomic.add(MCresult, 0, fun(x_tuple))
                    
    # load the device function
    exec(my_func, globals())

    dim = len(domain) 

    domain_range = np.array([(domain[j_dim][1] - domain[j_dim][0]) for j_dim in range(dim)], dtype=np.float64)
    domain = np.array(domain, dtype=np.float64)

    # Configure the threads and blocks
    threadsperblock = 32
    blockspergrid = (num_points + (threadsperblock - 1)) // threadsperblock

    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.sample(range(0,100000),1)[0])

    rng_states = cuda.to_device(rng_states)
    
    MCresult = cuda.to_device([0.])

    # Start the kernel
    integration_kernel[blockspergrid, threadsperblock](MCresult, domain, domain_range, rng_states, num_points)
    
    # volumn of the domain
    volumn = np.prod(domain_range)
    
    MCresult = MCresult.copy_to_host()

    result = volumn * MCresult / num_points

    
    return result


# In[165]:


def string_manipulation(funs):
    """
    given funcs and convert it into the required form"""
    
    temp =     """ 
import math
# define functions
    
@cuda.jit(device=True)
def fun(x):
    return """

    return [temp+i for i in funs]

class MCintegral_multi_function():
    def __init__(self, my_funcs = None, domains = None, head_node_address = None,
                 num_points = 20000):
        """
        Params
        ======
        my_funcs: list of functions
        domains: list of domains corresponding to each function
        num_points: number of sampled points for each function
        head_node_address: ray address"""
        
        self.funcs = string_manipulation(my_funcs)
        self.num_points = num_points
        self.domains = np.array(domains)
        self.num_funcs = len(self.funcs)
        
        # specify head node address
        if head_node_address == None:
            raise AssertionError("You must provide head node address with port.")
            
        ray.shutdown()
        ray.init(redis_address = head_node_address)
        
        # specify available number of gpus
        self.num_gpus = int(ray.get(gpu_nums.remote()))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print("Detected total number of GPUs: {}".format(self.num_gpus))
        
        print("Evaluating, please wait...")
        
    def evaluate(self):

        # result for all functions, len=len(self.my_funcs)
        MCresult = []

        # integrate each function parallelly on gpus
        for i_func in range(self.num_funcs):
            
            domain = self.domains[i_func]
            my_func = self.funcs[i_func]
            num_points = self.num_points
            
            # distributed calculation
            result = MCkernel.remote(domain, my_func, num_points)
            
            MCresult.append(result)
        
        # get data back to head node
        MCresult = ray.get(MCresult)
        
        return MCresult


# In[ ]:




