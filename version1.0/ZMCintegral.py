
# coding: utf-8

# In[117]:


'''
Coded by ZHANG Junjie (University of Science and Technology of China) in 09/2018.

This program is free: you can redistribute it and/or modify it under the terms of 
the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

The program requires python tensorflow and numpy to be pre-installed in your 
GPU-supported computer. 

'''
ZMCIntegral_VERSION='1.0'

import math
import numpy as np
import os,sys
import multiprocessing

# detect if GPU is available on the computer
def is_gpu_available(cuda_only=True):
    
    from tensorflow.python.client import device_lib as _device_lib
    
    if cuda_only:
        gpu_available=[int(x.name[-1]) for x in _device_lib.list_local_devices() if (x.device_type == 'GPU')]
        np.save(os.getcwd()+'/multi_temp/gpu_available', gpu_available)
    else:
        gpu_available=[int(x.name[-1]) for x in _device_lib.list_local_devices() if (x.device_type == 'GPU' or x.device_type == 'SYCL')]
        np.save(os.getcwd()+'/multi_temp/gpu_available', gpu_available)
        
def convert_1d_to_nd(one_d, dim, system_digit):
    '''convert one_d number to n_d of arbitrary systems
    system_digit: int, 2 means binary, 10 means the usual system'''
    import numpy as np
    import math
    
    temp_point=np.zeros(dim)
    for i_dim in range(dim):
        temp_i_one_d=one_d
        for temp_dim in range(dim):
            temp_i_one_d=temp_i_one_d-temp_point[temp_dim]*(system_digit**(dim-temp_dim-1))
        temp_point[i_dim]=math.floor(temp_i_one_d/(system_digit**(dim-i_dim-1)))
    return temp_point

def MCkernel(n_batches,domain,n_chunk_x,dim,my_func,n_grid_x_one_chunk,n_chunk,batch_size,i_batch):
    '''multiprocessing MC integration on different GPU
    n_batches: number of available GPUs
    domain: domain of the integral [[a,b],...]
    n_chunk_x: number of chunks along one dimension
    dim: dimensional of the integral
    my_func: user defined function
    n_grid_x_one_chunk: number of grid along one dimension in each chunk
    n_chunk: total number of chunks
    '''
    import tensorflow as tf
    from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE
    import os,sys
    
    def switch_to(mode):
        ctx = context()._eager_context
        ctx.mode = mode
        ctx.is_eager = mode == EAGER_MODE
    switch_to(EAGER_MODE)
    assert tf.executing_eagerly()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_batch)
    
    MCresult = 0.
    for i_chunk in range(batch_size):
        chunk_id = i_chunk + i_batch*batch_size
        if chunk_id<n_chunk:
            chunk_id_d_dim = convert_1d_to_nd(chunk_id,dim,n_chunk_x)
             
            domain_range=np.array([(domain[idim][1]-domain[idim][0])/n_chunk_x for idim in range(dim)],dtype=np.float32)
            domain_left=np.array([domain[idim][0]+chunk_id_d_dim[idim]*domain_range[idim] for idim in range(dim)],dtype=np.float32)
                
            dr_tensor=tf.expand_dims(domain_range,1)
            dl_tensor=tf.expand_dims(domain_left,1)
                    
            # random variables of sampling points
            random_domain_values = tf.add(tf.multiply(tf.random_uniform([dim,n_grid_x_one_chunk**dim],dtype=tf.float32),dr_tensor),dl_tensor)
            random_domain_values = list(map(tf.squeeze,tf.split(random_domain_values,dim,0),[0 for i in range(dim)]))
            
            # user defined function
            user_func = my_func(random_domain_values)
            
            # suppress singularities into 0.0
            user_func = tf.where(tf.is_nan(user_func), tf.zeros_like(user_func,dtype=tf.float32), user_func)
            user_func = tf.where(tf.is_inf(user_func), tf.zeros_like(user_func,dtype=tf.float32), user_func)
            
            # monte carlo result in this small chunk
            MCresult += tf.scalar_mul(np.prod(domain_range,dtype=np.float32),tf.reduce_mean(user_func,axis=-1)).numpy()
             
    return MCresult  

class MCintegral():
    '''
    my_func: user defined multidimansional function, type:function
    domain: integration domain, type:list/numpy_array
    available_GPU: list of available gpus, type: list, Default:All GPUs detected
    '''
    
    def __init__(self,my_func=None,domain=None,available_GPU=None):
        
        folder = os.getcwd()+'/multi_temp/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # clean temp file
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # check gpu condition
        p= multiprocessing.Process(target=is_gpu_available)
        p.daemon = True
        p.start()
        p.join()
        
        available_GPU=np.load(os.getcwd()+'/multi_temp/gpu_available.npy')
        if len(available_GPU) == 0:
            raise AssertionError("Your computer does not support GPU calculation.")
        
        self.available_GPU=available_GPU
        self.my_func,self.domain,self.dim,self.n_grid,self.chunk_size,self.n_batches =             self.initial(my_func,domain)
        
        self.MCresult = self.evaluate()
    
    def evaluate(self):
        '''Monte Carlo integration.'''
        
        dim=self.dim
        
        # n_grid_x: number of boxes along one dimension
        n_grid_x=int(np.round(self.n_grid**(1/self.dim)))
        
        # n_grid_x_one_chunk: number of boxes in one chunk along one dimension
        n_grid_x_one_chunk=int(np.round(self.chunk_size**(1/self.dim)))
        
        # n_chunk: total number of chunks
        n_chunk=int(np.round((self.n_grid/self.chunk_size)))
        
        # n_chunk_x: number of chunks along one dimension
        n_chunk_x=int(np.round(((self.n_grid)**(1/self.dim))/(self.chunk_size**(1/self.dim))))
        
        # batch_size
        batch_size=int(np.ceil(n_chunk/self.n_batches))
        
        p={}
        for i_batch in range(self.n_batches):
            def multi_processing():
                result=MCkernel(self.n_batches,self.domain,n_chunk_x,dim,self.my_func,n_grid_x_one_chunk,n_chunk,batch_size,i_batch)
                np.save(os.getcwd()+'/multi_temp/result'+str(i_batch), result)
                
            # start multi-processing to allocate     
            p[i_batch] = multiprocessing.Process(target=multi_processing)
            p[i_batch].daemon = True
            p[i_batch].start()
            
        for i_batch in range(self.n_batches):   
            p[i_batch].join()
                
        MCresult=0.
        for i_batch in range(self.n_batches): 
            MCresult+=np.load(os.getcwd()+'/multi_temp/result'+str(i_batch)+'.npy')
            
        return MCresult
    
    def initial(self,my_func,domain):
        '''Return proper initial consitions:
        dim: number of free variables, int,
        chunk_size: number of samplings in each chunk,int,
        n_grid: total number of d-dimensional samplings,int
        n_batches: seperate data into n_batches parts, int,
        '''
        
        dim=len(domain)
        if my_func==None:
            raise AssertionError("Invalid input function")
        elif domain==None or type(domain)==int:
            raise AssertionError("Please enter a correct domain")
            
        if dim==None:
            dim=1
        else:
            if type(dim)!=int:
                raise AssertionError("dim must be an int")
        
        # chunk_size: one batch contains many chunks
        if dim==1:
            n_grid=4194304
            chunk_size=65536
        elif dim==2:
            n_grid=(32768)**2
            chunk_size=(4096)**2
        elif dim==3:
            n_grid=(1024)**3
            chunk_size=(256)**3
        elif dim==4:
            n_grid=(192)**4
            chunk_size=(64)**4
        elif dim==5:
            n_grid=(64)**5
            chunk_size=(32)**5
        elif dim==6:
            n_grid=(32)**6
            chunk_size=(16)**6
        elif dim==7:
            n_grid=(20)**7
            chunk_size=(10)**7
        elif dim==8:
            n_grid=(14)**8
            chunk_size=(7)**8
        elif dim==9:
            n_grid=(10)**9
            chunk_size=(5)**9
        elif dim==10:
            n_grid=(8)**dim
            chunk_size=(4)**dim
        elif dim==11:
            n_grid=(6)**dim
            chunk_size=(3)**dim
        else:
            n_grid=(4)**dim
            chunk_size=(2)**dim    
            
        if len(domain)!=dim:
            raise AssertionError("Domain is inconsistant with dimension")
        for temp in domain:
            if len(temp)!=2:
                raise AssertionError("Domain is incorrect")
            if temp[1]<temp[0]:
                raise AssertionError("Domain [a,b] should satisfy b>a")
        
        # n_batches: number of batches
        n_batches=len(self.available_GPU)
        
        return my_func,domain,dim,n_grid,chunk_size,n_batches

