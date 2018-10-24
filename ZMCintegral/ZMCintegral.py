
# coding: utf-8

# In[46]:


'''
Coded by ZHANG Junjie (University of Science and Technology of China) in 10/2018.

This program is free: you can redistribute it and/or modify it under the terms of 
the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

The program requires python tensorflow and numpy to be pre-installed in your 
GPU-supported computer. 

'''
ZMCIntegral_VERSION = '2.2'

import tensorflow as tf
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE
import os,sys
import math
import numpy as np
import multiprocessing
        
# detect if GPU is available on the computer
def is_gpu_available(cuda_only = True):
    
    from tensorflow.python.client import device_lib as _device_lib
    
    if cuda_only:
        gpu_available=[int(x.name[-1]) for x in _device_lib.list_local_devices() if (x.device_type == 'GPU')]
        np.save(os.getcwd()+'/multi_temp/gpu_available', gpu_available)
    else:
        gpu_available=[int(x.name[-1]) for x in _device_lib.list_local_devices() if (x.device_type == 'GPU' or x.device_type == 'SYCL')]
        np.save(os.getcwd()+'/multi_temp/gpu_available', gpu_available)
        


class MCintegral():
    
    def __init__(self, my_func = None, domain = None, available_GPU = None, num_trials = 5, depth = None, sigma_multiplication = 4,method=None):

        '''
        Parameters:
            my_func: user defined multidimensional function, type:function
            domain: integration domain, type:list/numpy_array, eg [[0,1]] or [[0,1],[0,1]]
            available_GPU: list of available gpus, type: list, Default: All GPUs detected, eg [0,1,2,3]
            num_trial: number of trials, type:int, Default:5
            depth: search depth, type:int, Default:2
            sigma_multiplication: recalculate the grid if `stddev` larger than `sigma_mean + sigma_multiplication * sigma`, type:float, Default:4
        '''
        
        # choose eager mode
        def switch_to(mode):
            ctx = context()._eager_context
            ctx.mode = mode
            ctx.is_eager = (mode == EAGER_MODE)
        # set to eager mode
        switch_to(EAGER_MODE)
        assert tf.executing_eagerly()
        
        # clean temp file
        self.clean_temp()
        
        if available_GPU == None:
            # check gpu condition
            p = multiprocessing.Process(target = is_gpu_available)
            p.daemon = True
            p.start()
            p.join()
            
            available_GPU = np.load(os.getcwd() + '/multi_temp/gpu_available.npy')
        
        if len(available_GPU) == 0:
            raise AssertionError("Your computer does not support GPU calculation.")
            
        if method == None or method == 'AdaptiveImportanceMC':
        
            # number of trials
            self.num_trials = num_trials
            
            # depth of the zooming search
            if depth==None:
                self.depth = 2
            else:
                self.depth = depth
             
            self.method = 'AdaptiveImportanceMC'
        
        if method == 'AverageDigging':
        
            # number of trials
            self.num_trials = 1
            
            # depth of the zooming search
            if depth==None:
                self.depth = 1
            else:
                self.depth = depth
                
            self.method='AverageDigging'
        
        
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
        if depth < self.depth:
            for chunk_id in large_std_chunk_id:
                # domain of this chunk
                domain_next_level = self.chunk_domian(chunk_id, domain)
                # iteration
                MCresult_chunks[chunk_id],MCresult_std_chunks[chunk_id] = self.importance_sampling_iteration(domain_next_level, depth)
        
        # Stop digging if there are no more large stddev chunk
        if len(large_std_chunk_id) == 0:
            return np.sum(MCresult_chunks,0), np.max(MCresult_std_chunks)

        return np.sum(MCresult_chunks,0), np.max(MCresult_std_chunks)
    
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
        
        MCresult, MCresult_std = np.concatenate(MCresult), np.concatenate(MCresult_std)
        
        # find out the index of chunks that have very large stddev
        if len(np.shape(MCresult))==1:
            threshold = np.mean(MCresult_std) + self.sigma_multiplication * np.std(MCresult_std)
            large_std_chunk_id = np.where(MCresult_std >= threshold)[0]
            return MCresult, large_std_chunk_id, MCresult_std
        else:
            MCresult_std = np.transpose(MCresult_std)
            threshold = np.mean(MCresult_std,-1) + self.sigma_multiplication * np.std(MCresult_std,-1)
            large_std_chunk_id = np.unique(np.concatenate([np.where(MCresult_std[i] >= threshold[i])[0] for i in range(len(MCresult_std))]))
            return MCresult, large_std_chunk_id, np.transpose(MCresult_std)
        
        
        
    
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
            self.chunk_size_x = 65536
            
        elif self.dim == 2:
            self.chunk_size_x = 4096
            
        elif self.dim == 3:
            self.chunk_size_x = 256

        elif self.dim == 4:
            self.chunk_size_x = 64
            
        elif self.dim == 5:
            self.chunk_size_x = 24
            
        elif self.dim == 6:
            self.chunk_size_x = 10
            
        elif self.dim == 7:
            self.chunk_size_x = 8
            
        elif self.dim == 8:
            self.chunk_size_x = 6
            
        elif self.dim == 9:
            self.chunk_size_x = 5
            
        elif self.dim == 10:
            self.chunk_size_x = 4
            
        elif self.dim == 11:
            self.chunk_size_x = 3
            
        else:
            self.chunk_size_x = 2
        
        self.chunk_size_multiplier = int(math.floor((len(self.available_GPU)*192)**(1/self.dim)))
        
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

        chunk_id_d_dim = self.convert_1d_to_nd(chunk_id, self.dim, self.n_chunk_x)
        domain_range = np.array([(original_domain[idim][1] - original_domain[idim][0]) / self.n_chunk_x for idim in range(self.dim)], dtype=np.float32)
        domain_left = np.array([original_domain[idim][0] + chunk_id_d_dim[idim] * domain_range[idim] for idim in range(self.dim)], dtype=np.float32)
        current_domain = [[domain_left[i], domain_left[i] + domain_range[i]] for i in range(self.dim)]
        return current_domain
    
    def convert_1d_to_nd(self, one_d, dim, system_digit):

        '''
        Function:
            convert `one_d` index to `n_d` index of arbitrary systems
        Parameters:
            one_d: current index in the whole 1 dimension sequence, type:int.
            dim: the real system dimension, type:int.
            system_digit: the length in one dimension of the real system, type:int.
        '''

        temp_point = np.zeros(dim)
        for i_dim in range(dim):
            temp_i_one_d = one_d
            for temp_dim in range(dim):
                temp_i_one_d -= temp_point[temp_dim] * (system_digit**(dim-temp_dim-1))
            temp_point[i_dim] = math.floor(temp_i_one_d / (system_digit**(dim-i_dim-1)))
        return temp_point
    
    def MCkernel(self, domain, i_batch):

        '''
        Function:
            multiprocessing Monte Carlo integration on specific GPU
        Parameters:
            domain: domain of the integral, eg: [[a,b],[c,d],...].
            i_batch: the index of current GPU, type:int.
        '''

        MCresult = []
        for i_chunk in range(self.batch_size):
            chunk_id = i_chunk + i_batch * self.batch_size
            if chunk_id < self.n_chunk:
                chunk_id_d_dim = self.convert_1d_to_nd(chunk_id, self.dim, self.n_chunk_x)
             
                domain_range = np.array([(domain[idim][1] - domain[idim][0]) / self.n_chunk_x for idim in range(self.dim)], dtype=np.float32)
                domain_left = np.array([domain[idim][0] + chunk_id_d_dim[idim] * domain_range[idim] for idim in range(self.dim)], dtype=np.float32)
                    
                if self.method=='AdaptiveImportanceMC':
                    # random variables of sampling points
                    random_domain_values = [tf.random_uniform([self.chunk_size], minval=domain_left[i_dim],                                                          maxval=domain_left[i_dim]+domain_range[i_dim],dtype=tf.float32)                                            for i_dim in range(self.dim)]
                
                elif self.method=='AverageDigging':
                    # sampling specified points
                    domain_temp = [tf.range(start=0,limit=self.chunk_size_x,delta=1,dtype=tf.float32)/self.chunk_size_x*domain_range[i_dim]                                            +domain_left[i_dim]+domain_range[i_dim]*0.5/(self.chunk_size_x)                                            for i_dim in range(self.dim)]
                    meshed_domain = tf.meshgrid(*domain_temp)
                    random_domain_values = [tf.reshape(meshed_domain[i_dim],[self.chunk_size,]) for i_dim in range(self.dim)]
                    
                # user defined function, tensor calculation
                user_func = self.my_func(random_domain_values)
            
                # suppress singularities into 0.0
                user_func = tf.where(tf.is_nan(user_func), tf.zeros_like(user_func, dtype=tf.float32), user_func)
                user_func = tf.where(tf.is_inf(user_func), tf.zeros_like(user_func, dtype=tf.float32), user_func)
            
                # monte carlo result in this small chunk
                MCresult.append(tf.scalar_mul(np.prod(domain_range, dtype=np.float32), tf.reduce_mean(user_func, -1)).numpy())
           
        return np.array(MCresult) 

