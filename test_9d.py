import ZMCintegral

import time
a=time.time()
# user defined function
fun = """ 
import math
# define a device function that should be used by cuda kernel
@cuda.jit(device=True)
def fun(x):
    return ((1/math.sqrt(2 * math.pi * 0.0001))**9) * math.exp(-(x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2+x[6]**2+x[7]**2+x[8]**2)/(2*0.0001))
"""
depth = 4
sigma_multiplier = 5
num_trials = 5
num_chunks_in_one_dimension = 3

MC = ZMCintegral.MCintegral(my_func = fun, domain = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]], 
                head_node_address = "210.45.78.43:6789",depth = depth, sigma_multiplier = sigma_multiplier, 
                num_trials = num_trials, num_chunks_in_one_dimension = num_chunks_in_one_dimension)

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
print(time.time()-a)