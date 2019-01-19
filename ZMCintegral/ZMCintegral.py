'''
Coded by ZHANG Junjie (University of Science and Technology of China) in 01/2019.

This program is free: you can redistribute it and/or modify it under the terms of 
the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

The program requires python numba to be pre-installed in your 
GPU-supported computer. 

'''
ZMCIntegral_VERSION = '3.0'

import math
import numpy as np
import multiprocessing
from numba import cuda
import numba as nb
import random
import os
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

class MCintegral():
    
    def __init__(self, my_func = None, domain = None, available_GPU = None, num_trials = 2, depth = 2, sigma_multiplication = 5):

        '''
        Parameters:
            my_func: user defined multidimensional function, type:function
            domain: integration domain, type:list/numpy_array, eg [[0,1]] or [[0,1],[0,1]]
            available_GPU: list of available gpus, type: list, Default: All GPUs detected, eg [0,1,2,3]
            num_trial: number of trials, type:int, Default:2
            depth: search depth, type:int, Default:2
            sigma_multiplication: recalculate the grid if `stddev` larger than `sigma_mean + sigma_multiplication * sigma`, type:float, Default:5
        '''
        
        # clean temp file
        self.clean_temp()
        
        if available_GPU == None:
            def is_gpu_available():
                np.save(os.getcwd()+'/multi_temp/gpu_available', [i for i in range(len(list(cuda.gpus)))])
                
            # check gpu condition
            p = multiprocessing.Process(target = is_gpu_available)
            p.daemon = True
            p.start()
            p.join()
            
            available_GPU =  np.load(os.getcwd() + '/multi_temp/gpu_available.npy')
        
        if len(available_GPU) == 0:
            raise AssertionError("Your computer does not support GPU calculation.")
            
        # number of trials
        self.num_trials = num_trials
            
        self.depth = depth

        # recalculate the grid if `stddev` larger than `sigma_mean + sigma_multiplication * sigma`
        self.sigma_multiplication = sigma_multiplication
        
        # set up initial conditions
        self.available_GPU = available_GPU

        # initialize the preparing integrated function depend on its domain dimension
        self.initial(my_func, domain)
        
        # initial domain
        self.initial_domain = domain
       
    
    def evaluate(self):
        self.configure_chunks()
        MCresult = self.importance_sampling_iteration(self.initial_domain, 0)
        
        return MCresult
    
    def importance_sampling_iteration(self, domain, depth):
        depth += 1
        MCresult_chunks, large_std_chunk_id, MCresult_std_chunks = self.MCevaluate(domain)
        print('{} hypercube(s) need(s) to be recalculated, to save time, try drastically increasing sigma_multiplication.'.format(len(large_std_chunk_id)))
        if depth < self.depth:
            for chunk_id in large_std_chunk_id:
                # domain of this chunk
                domain_next_level = self.chunk_domian(chunk_id, domain)
                
                # iteration
                MCresult_chunks[chunk_id],MCresult_std_chunks[chunk_id] = self.importance_sampling_iteration(domain_next_level, depth)
                
        # Stop digging if there are no more large stddev chunk
        if len(large_std_chunk_id) == 0:
            return np.sum(MCresult_chunks,0), np.sqrt(np.sum(MCresult_std_chunks**2))

        return np.sum(MCresult_chunks,0), np.sqrt(np.sum(MCresult_std_chunks**2))
    
    def MCevaluate(self, domain):

        '''
        Monte Carlo integration.
        Parameters:
            domain: the integration domain, type:list or numpy_array.
        '''
        
        p={}
        for i_batch in range(self.n_batches):
            def multi_processing():
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(i_batch)
                result = []
                for trial in range(self.num_trials):
                    result.append(self.MCkernel(domain, i_batch))
                    
                result = np.array(result)
                std_result = np.std(result,0)
                mean_result = np.mean(result,0)
                np.save(os.getcwd()+'/multi_temp/result'+str(i_batch), np.array(mean_result))
                np.save(os.getcwd()+'/multi_temp/result_std'+str(i_batch), np.array(std_result))
                
            # start multi-processing to allocate     
            p[i_batch] = multiprocessing.Process(target = multi_processing)
            p[i_batch].daemon = True
            p[i_batch].start()
            
        for i_batch in range(self.n_batches):   
            p[i_batch].join()
                
        MCresult = []
        MCresult_std = []
        for i_batch in range(self.n_batches): 
            MCresult.append(np.load(os.getcwd()+'/multi_temp/result'+str(i_batch)+'.npy'))
            MCresult_std.append(np.load(os.getcwd()+'/multi_temp/result_std'+str(i_batch)+'.npy'))
        
        MCresult, MCresult_std = np.concatenate(MCresult), np.array(MCresult_std)
    
        # find out the index of chunks that have very large stddev
        threshold = np.mean(MCresult_std) + self.sigma_multiplication * np.std(MCresult_std)
        len_std = len(MCresult_std[0])
        large_std_chunk_id = np.concatenate([np.nonzero(MCresult_std[i] >= threshold)[0] + i*len_std for i in range(len(MCresult_std))])
        return MCresult, large_std_chunk_id, np.concatenate(MCresult_std)
        
        
    def initial(self, my_func, domain):
        '''
        To obtain proper initial conditions:
            self.dim: number of free variables, type:int,
            self.chunk_size: number of samplings in each chunk, type:int
            self.n_grid: total number of d-dimensional samplings, type:int
            self.n_batches: seperate data into n_batches parts, type:int
        Parameters:
            my_func: user defined multidimensional function, type:function
            domain: integration domain, type:list/numpy_array, eg [[0,1]] or [[0,1],[0,1]]
        '''    

        # detect if enter a function             
        if my_func == None:
            raise AssertionError("Invalid input function")
        # the preparing integrated function
        self.my_func = my_func

        # detect if domain is in right form
        if domain == None:
            raise AssertionError("Please enter a domain")
        for temp in domain:
            if len(temp) != 2:
                raise AssertionError("Domain is incorrect")
            if temp[1] < temp[0]:
                raise AssertionError("Domain [a,b] should satisfy b>a")
                
        # integrating dimension
        self.dim = len(domain)
        
        # get `total sampling number` and `sampling number in one chunk` depend on dimension of integral       
        if self.dim == 1:
            self.chunk_size_x = 10000
            self.n_one_gpu = 99999
            
        elif self.dim == 2:
            self.chunk_size_x = 100
            self.n_one_gpu = 999
            
        elif self.dim == 3:
            self.chunk_size_x = 35
            self.n_one_gpu = 99

        elif self.dim == 4:
            self.chunk_size_x = 10
            self.n_one_gpu = 20
            
        elif self.dim == 5:
            self.chunk_size_x = 7
            self.n_one_gpu = 13
            
        elif self.dim == 6:
            self.chunk_size_x = 6
            self.n_one_gpu = 7
            
        elif self.dim == 7:
            self.chunk_size_x = 4
            self.n_one_gpu = 6
            
        elif self.dim == 8:
            self.chunk_size_x = 3
            self.n_one_gpu = 5
            
        elif self.dim == 9:
            self.chunk_size_x = 3
            self.n_one_gpu = 4
            
        elif self.dim == 10:
            self.chunk_size_x = 3
            self.n_one_gpu = 3
            
        elif self.dim == 11:
            self.chunk_size_x = 2
            self.n_one_gpu = 3
            
        elif self.dim == 12:
            self.chunk_size_x = 2
            self.n_one_gpu = 3
           
        elif self.dim == 13:
            self.chunk_size_x = 2
            self.n_one_gpu = 2
            
        elif self.dim == 14:
            self.chunk_size_x = 2
            self.n_one_gpu = 2
            
        elif self.dim == 15:
            self.chunk_size_x = 2
            self.n_one_gpu = 2
            
        elif self.dim == 16:
            self.chunk_size_x = 2
            self.n_one_gpu = 2
            
        else:
            self.chunk_size_x = 1
            self.n_one_gpu = 2
        
        n_gpu = len(self.available_GPU)
        self.chunk_size_multiplier = math.floor((n_gpu*(self.n_one_gpu**self.dim))**(1/self.dim))
        
    def configure_chunks(self):
        '''receieve self.dim, self.n_grid and self.chunk_size'''
        
        '''
            below, `int(np.round())` can make sure you got the exact number, 
            eg: in Python, you may get 7.99999 from 64^(1/2)
        '''
        
        self.chunk_size = self.chunk_size_x**self.dim
        self.n_grid = (self.chunk_size_x*self.chunk_size_multiplier)**self.dim
        
        # number of samplings in one chunk along one dimension
        self.n_grid_x_one_chunk = int(np.round(self.chunk_size**(1/self.dim)))
        
        # number of chunks
        self.n_chunk = int(np.round(self.n_grid/self.chunk_size))
        
        # number of samplings along one dimension
        self.n_grid_x = int(np.round(self.n_grid**(1/self.dim)))
        
        # number of chunks along one dimension
        self.n_chunk_x = int(np.round(self.n_chunk**(1/self.dim)))
        
        # number of batches
        self.n_batches = min([len(self.available_GPU), self.n_chunk])
        
        # batch_size
        self.batch_size = int(np.ceil(self.n_chunk/self.n_batches))

    def clean_temp(self):
        folder = os.getcwd()+'/multi_temp/'
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # clean temp file
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
    def chunk_domian(self, chunk_id, original_domain):

        '''
        Return:
            domain of integration in this chunk.
        Parameters:
            chunk_id: current chunk id, type:int.
            original_domain: the domain of the previous original integration.
        '''
        
        chunk_id_d_dim = np.unravel_index(chunk_id, [self.n_chunk_x for _ in range(self.dim)])
        domain_range = np.array([(original_domain[idim][1] - original_domain[idim][0]) / self.n_chunk_x for idim in range(self.dim)], dtype=np.float64)
        domain_left = np.array([original_domain[idim][0] + chunk_id_d_dim[idim] * domain_range[idim] for idim in range(self.dim)], dtype=np.float64)
        current_domain = [[domain_left[i], domain_left[i] + domain_range[i]] for i in range(self.dim)]
        return current_domain
    
    def MCkernel(self, domain, i_batch):

        '''
        Function:
            multiprocessing Monte Carlo integration on specific GPU
        Parameters:
            domain: domain of the integral, eg: [[a,b],[c,d],...].
            i_batch: the index of current GPU, type:int.
        '''
        fun = self.my_func
        dim = self.dim
        batch_size = self.batch_size
        MCresult = cuda.device_array(batch_size,dtype=np.float64)
        n_chunk = self.n_chunk
        chunk_size = self.chunk_size
        num_loops = chunk_size * batch_size
        n_chunk_x = self.n_chunk_x
        domain_range = np.array([(domain[j_dim][1] - domain[j_dim][0]) / n_chunk_x for j_dim in range(dim)],                                dtype = np.float64)
        domain = np.array(domain,dtype=np.float64)
        
        # change one dimensional index into 
        @cuda.jit(device=True)
        def oneD_to_nD(num_of_points_in_each_dim,new_i,digit_store):
            j_dim_index = 0
            a1 = new_i//num_of_points_in_each_dim
            digit_store[j_dim_index] = new_i%num_of_points_in_each_dim
            
            # convert to n-dim index
            for j_dim in range(dim):
                j_dim_index+=1
                if a1 != 0.:
                    digit_store[j_dim+1] = a1%num_of_points_in_each_dim
                    a1 = a1//num_of_points_in_each_dim
            
        @cuda.jit
        def integration_kernel(num_loops,                               MCresult,                               chunk_size,                               n_chunk_x,                               domain,                               domain_range,                               batch_size,                               i_batch,                               rng_states,                               n_chunk):
            
            thread_id = cuda.grid(1)
            if thread_id < batch_size:
                chunk_id = thread_id + i_batch * batch_size
            
                if chunk_id < n_chunk:
            
                    # digit_store: local digits index for each thread
                    digit_store = cuda.local.array(shape=dim, dtype=nb.int64)
                    for i_temp in range(dim):
                        digit_store[i_temp] = 0
                    
                    # convert one_d index to dim_d index
                    # result will be stored in digit_store
                    oneD_to_nD(n_chunk_x,chunk_id,digit_store)
            
                    # specisify the local domain
                    domain_left = cuda.local.array(dim, dtype=nb.float64)
                    for j_dim in range(dim):
                        domain_left[j_dim] = domain[j_dim][0] + digit_store[j_dim] * domain_range[j_dim]
            
                    for i_sample in range(chunk_size):
                        # x_tuple: local axis values for each thread
                        x_tuple = cuda.local.array(dim, dtype=nb.float64)
                
                        for j_dim in range(dim):
                            x_tuple[j_dim] = xoroshiro128p_uniform_float64(rng_states, thread_id)                                                            *domain_range[j_dim] + domain_left[j_dim]
                
                        # feed in values to user defined function
                        cuda.atomic.add(MCresult, thread_id, fun(x_tuple))
               
        # Configure the blocks
        threadsperblock = 16
        blockspergrid = (batch_size + (threadsperblock - 1)) // threadsperblock
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid,                                                  seed=random.sample(range(0,1000),1)[0])
                    
        # Start the kernel 
        integration_kernel[blockspergrid, threadsperblock](num_loops,                                                           MCresult,                                                           chunk_size,                                                           n_chunk_x,                                                           domain,                                                           domain_range,                                                           batch_size,                                                           i_batch,                                                           rng_states,                                                           n_chunk)
        
        # volumn of the domain
        volumn = np.prod(domain_range)/chunk_size
        
        MCresult = MCresult.copy_to_host()
        MCresult = volumn*MCresult
        
        return MCresult
    

