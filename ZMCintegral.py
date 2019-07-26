'''
Coded by Jun-Jie Zhang (University of Science and Technology of China) in 06/2019
and checked by Hong-Zhong Wu(University of Science and Technology of China).

This program is free: you can redistribute it and/or modify it under the terms of 
the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

The program requires python numba and ray to be pre-installed in your 
GPU-supported computer. 

'''
ZMCIntegral_VERSION = '4.0.'

import math
import numpy as np
from numba import cuda
import numba as nb
import random
import os
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import ray

@ray.remote(num_gpus=1)
def gpu_nums():
    # return all detected available gpus, int
    detected_gpu_id = ray.cluster_resources()['GPU']
    return detected_gpu_id

def clean_temp():
    folder = os.getcwd()+'/multi_temp/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # clean temp file
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

@ray.remote(num_gpus=1)
def MCkernel(domain, i_batch, my_func, dim, batch_size, num_chunks,
             num_points_in_one_chunk, num_chunks_in_one_dimension, num_trials):

    '''
    Function:
        do Monte-Carlo integration on specific GPU device with uniform sampling on 
        every given chunk domain that the specific GPU device will process. 
        Remember that the current GPU device will process as possible as 'batch_size'
        chunks in each time when chunks' allocation happens.
    Parameters:
        @domain: domain of the integral, eg: [[a,b],[c,d],...].
        @i_batch: current batch index, type:int.
        @my_func: user defined function, should be in sting format and the integrand's name must be 'fun'.
        @dim: integration dimension.
        @batch_size: number of chunks in one allocated process.
        @num_chunks: number of total chunks.
        @num_points_in_one_chunk: number of samples in one chunk.
        @num_chunks_in_one_dimension: number of chunks in every dimension.
        @num_trials: the number of independent trials for current chunk.
    '''
    # have a look at current batch index and node ip.
    #print(i_batch)
    # execute user defined function as global variable
    exec(my_func, globals()) 
    
    # result for accumulating different trials
    trial_result = np.zeros([num_trials, batch_size])
    # get the small chunks' range, later this array will be used to calculate the volumn of specific integration domain
    domain_range = np.array([(domain[j_dim][1] - domain[j_dim][0]) / num_chunks_in_one_dimension for j_dim in range(dim)],
                            dtype=np.float64)
    # incase the original input is a list like type
    domain = np.array(domain, dtype=np.float64)

    # this function convert one dimensional index to n dimensional index locally.
    @cuda.jit(device=True)
    def oneD_to_nD(digit_x, oneD_idx, digit_store):
        '''
        Principle used here:
            oneD_idx = c0*(digit_x**0) +c1*(digit_x**1)+c2*(digit_x**2)+c3*(digit_x**3)+...
        Parameters:
            @digit_x: the scaled number in one dimension
            @oneD_idx: current one dimensional index
            @digit_store: store the converted n dimensional index
        '''
        j_dim_index = 0
        digit_store[j_dim_index] = oneD_idx % digit_x
        a1 = oneD_idx // digit_x
        
        # convert to n-dim index
        for j_dim in range(dim):
            j_dim_index += 1
            if a1 != 0.:
                digit_store[j_dim + 1] = a1 % digit_x
                a1 = a1 // digit_x

    @cuda.jit
    def integration_kernel(MCresult, num_points_in_one_chunk, num_chunks_in_one_dimension,
                           domain, domain_range, batch_size, i_batch, rng_states, num_chunks):

        thread_id = cuda.grid(1)
        if thread_id < batch_size:
            chunk_id = thread_id + i_batch * batch_size

            if chunk_id < num_chunks:

                # local digits index for each thread
                digit_store = cuda.local.array(shape=dim, dtype=nb.int64)
                for i_temp in range(dim):
                    digit_store[i_temp] = 0

                # convert one_dim index to n_dim index
                # result will be stored in digit_store
                oneD_to_nD(num_chunks_in_one_dimension, chunk_id, digit_store)

                # specify the local domain
                domain_left = cuda.local.array(shape=dim, dtype=nb.float64)
                for j_dim in range(dim):
                    domain_left[j_dim] = domain[j_dim][0] + digit_store[j_dim] * domain_range[j_dim]

                for i_sample in range(num_points_in_one_chunk):
                    # x_tuple: local axis values for each thread
                    x_tuple = cuda.local.array(shape=dim, dtype=nb.float64)

                    for j_dim in range(dim):
                        x_tuple[j_dim] = xoroshiro128p_uniform_float64(rng_states, thread_id) * domain_range[j_dim] + domain_left[j_dim]

                    # feed in values to user defined function, 
                    # and add all points' corresponding results in one chunk
                    cuda.atomic.add(MCresult, thread_id, fun(x_tuple))

    # Configure the threads and blocks
    threadsperblock = 32
    blockspergrid = (batch_size + (threadsperblock - 1)) // threadsperblock

    for i_trial in range(num_trials):
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, 
                                             seed=random.sample(range(0,100000),1)[0])
        rng_states = cuda.to_device(rng_states)

        MCresult = cuda.device_array(batch_size, dtype=np.float64)
        # Start the kernel
        integration_kernel[blockspergrid, threadsperblock](MCresult, num_points_in_one_chunk, num_chunks_in_one_dimension,
                                                           domain, domain_range, batch_size, i_batch, rng_states, num_chunks)

        # volumn of the domain divided by number of points in one chunk
        volumn = np.prod(domain_range) / num_points_in_one_chunk
        
        MCresult = MCresult.copy_to_host()

        MCresult = volumn * MCresult

        trial_result[i_trial] = MCresult
        
    print("current batch: {0}, trial number: {1}".format(i_batch, num_trials))

    return trial_result


@ray.remote(num_gpus=1)
def MCkernel_mean(trial_result):
    return np.mean(trial_result,0)

@ray.remote(num_gpus=1)
def MCkernel_std(trial_result):
    return np.std(trial_result,0)


class MCintegral():
    def __init__(self, my_func = None, domain = None,head_node_address = None,
                 num_trials = 5, depth = 2, sigma_multiplier = 5, num_points_in_one_chunk = 10000,
                 num_chunks_in_one_dimension = 4, batch_size = 200000):

        '''
        Parameters:
            @my_func: user defined multidimensional function, type:string, the integrand must have name "fun"
            @domain: integration domain, type:list/numpy_array, eg [[0,1]] or [[0,1],[0,1]]
            @head_node_address: head node address with port, type: str 
            @num_trials: number of independent trials for every chunk, type:int, Default:2
            @depth: depth of heuristic tree search, type:int, Default:2
            @sigma_multiplier: recalculate the grid if `stddev` larger than `sigma_mean + sigma_multiplier * sigma`, 
                                    type:float, Default:5
            @num_points_in_one_chunk: number of samples in one chunk, type: int
            @num_chunks_in_one_dimension: number of chunks in every dimension.
            @batch_size: number of chunks in one allocated process.
        '''

        # clean temporary file
        clean_temp()
        
        # specify head node address
        if head_node_address == None:
            raise AssertionError("You must provide head node address with port.")
            
        ray.shutdown()
        ray.init(redis_address = head_node_address)

        # specify available number of gpus
        self.num_gpus = int(ray.get(gpu_nums.remote()))
        print("total number of GPUs: ", self.num_gpus)
      
        # number of trials
        self.num_trials = num_trials
        
        # depth of the digging
        self.depth = depth

        # recalculate the grid if `stddev` larger than `sigma_mean + sigma_multiplier * sigma`
        self.sigma_multiplier = sigma_multiplier
        
        # detect if domain is in right form
        if domain == None:
            raise AssertionError("Please enter a domain")
        for temp in domain:
            if len(temp) != 2:
                raise AssertionError("Domain is incorrect")
            if temp[1] < temp[0]:
                raise AssertionError("Domain [a,b] should satisfy b>a")
                
        # initial domain
        self.initial_domain = domain
                
        # integrating dimension
        self.dim = len(domain)
        
        # user defined function
        self.my_func = my_func
        
        # number of chunks for each gpu to evaluate in each gpu allocation process
        # this is the number of threads
        self.batch_size = batch_size
        
        # number of sampling points in one chunk 
        # this number should not be very large otherwise the gpu will yield overflow
        self.num_points_in_one_chunk = num_points_in_one_chunk
            
        # number of chunks in one dimension
        self.num_chunks_in_one_dimension = num_chunks_in_one_dimension

        # number of chunks
        self.num_chunks = self.num_chunks_in_one_dimension ** self.dim
        
        # number of batches, i.e., the required total number of gpu allocation
        self.num_batches = self.num_chunks // self.batch_size + 1
        
    def evaluate(self):
        MCresult = self.stratified_sampling_iteration(self.initial_domain, 0)        
        return MCresult
    
    def stratified_sampling_iteration(self, domain, depth):
        depth += 1
        MCresult_chunks, MCresult_std_chunks, large_std_chunk_id = self.MCevaluate(domain)
        print('{} hypercube(s) need(s) to be recalculated, to save time, try increasing sigma_multiplier.'.format(len(large_std_chunk_id)))
        if depth < self.depth:
            for chunk_id in large_std_chunk_id:
                # domain of this chunk
                domain_next_level = self.chunk_domian(chunk_id, domain)
                
                # iteration
                MCresult_chunks[chunk_id], MCresult_std_chunks[chunk_id] = self.stratified_sampling_iteration(domain_next_level, depth)
                
        # Stop digging if there are no more large stddev chunk even the required digging depth is not reached
        if len(large_std_chunk_id) == 0:
            return np.sum(MCresult_chunks, 0), np.sqrt(np.sum(MCresult_std_chunks**2))

        return np.sum(MCresult_chunks, 0), np.sqrt(np.sum(MCresult_std_chunks**2))

    
    def MCevaluate(self, domain):

        '''
        Monte Carlo integration.
        Parameters:
            @domain: the integration domain, type:list or numpy_array.
        '''
        
        # result for accumulating data
        MCresult = []
        MCresult_std = []
        
        # loop through all gpus
        for i_batch in range(self.num_batches):
            
            # distribute calculation
            trial_result = MCkernel.remote(domain, i_batch, self.my_func, self.dim,
                                           self.batch_size, self.num_chunks, self.num_points_in_one_chunk,
                                           self.num_chunks_in_one_dimension, self.num_trials)
            
            MCresult.append(MCkernel_mean.remote(trial_result))
            MCresult_std.append(MCkernel_std.remote(trial_result))

        # get data back to head node
        MCresult = np.concatenate(ray.get(MCresult))
        MCresult_std = np.concatenate(ray.get(MCresult_std))
       
        # find out the index of chunks that have very large stddev
        threshold = np.mean(MCresult_std) + self.sigma_multiplier * np.std(MCresult_std)
        large_std_chunk_id = np.nonzero(MCresult_std >= threshold)[0]
       
        return MCresult, MCresult_std, large_std_chunk_id

    
    def chunk_domian(self, chunk_id, original_domain):

        '''
        Return:
            domain of integration in this chunk.
        Parameters:
            @chunk_id: current chunk id, type:int.
            @original_domain: the domain of the previous original integration.
        '''
        
        chunk_id_d_dim = np.unravel_index(chunk_id, [self.num_chunks_in_one_dimension for _ in range(self.dim)])
        domain_range = np.array([(original_domain[idim][1] - original_domain[idim][0]) / self.num_chunks_in_one_dimension for idim in range(self.dim)], dtype=np.float64)
        domain_left = np.array([original_domain[idim][0] + chunk_id_d_dim[idim] * domain_range[idim] for idim in range(self.dim)], dtype=np.float64)
        current_domain = [[domain_left[i], domain_left[i] + domain_range[i]] for i in range(self.dim)]
        return current_domain    