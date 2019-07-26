import ZMCintegral

import time
a=time.time()

# user defined function
fun = """ 
import math
# define a device function that should be used by cuda kernel
@cuda.jit(device=True)
def fun(x):
    return math.sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6])
"""

depth = 1
sigma_multiplier = 5
num_trials = 5
num_chunks_in_one_dimension = 12

MC = ZMCintegral.MCintegral(my_func = fun, domain = [[0,10],[0,10],[0,10],[0,10],[0,10],[0,10]], head_node_address = "210.45.78.43:6789",
                depth = depth, sigma_multiplier = sigma_multiplier, num_trials = num_trials,
                num_chunks_in_one_dimension = num_chunks_in_one_dimension)

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
print(time.time()-a)