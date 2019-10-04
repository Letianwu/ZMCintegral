# ZMCintegral

ZMCintegral (Numba backened) is an easy to use python package which uses Monte Carlo Evaluation Method to do numerical integrations on Multi-GPU devices. 
It supports integrations with up to 16 multi-variables, and it is capable of even more than 16 variables if time is not of the priori concern. 

> **To understand how ZMCintegral works, please refer to**

  **https://arxiv.org/pdf/1902.07916v2.pdf**
    
> **This new version supports parameter grid search, for this new functionality please refer to**

  **??????????**
    
ZMCintegral usually takes a few minutes to finish the task.

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Newest Features

  - Full flexibility of user defined functions
  - Multi-dimension integration
  - Multi-GPU supports
  - Stratified sampling
  - Heuristic tree search
  - Parameter grid search
  



## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Installation

To run ZMCintegral (Numba-Ray version), the following packages needs to be pre-installed:
  - Numba
  - Ray
  - cudatoolkit
```
$: conda install python=3.6
$: conda install numba
$: conda install cudatoolkit
$: pip install -U ray[debug]
```
ZMCintegral can be installed simply via
```
$: pip install ZMCintegral
```

#### How to use the package
First of all, prepare machines with Nvidia GPU devices. choose one of them as a head node:
```
# for head node
$: ray start --head --redis-port=6789 --num-cpus=10 --num-gpus=4
#for other nodes, here the redis-address is the ip of head node.
$: ray start --redis-address=210.45.78.43:6789 --num-cpus=5 --num-gpus=2
```
After that, you can try ZMCintegral.

#### - tuning parameters

The following four parameters can be tuned to fit special cases.

| parameter        | usage           | example           | default  |
|:-------------:|:-------------:|:-------------:|:-----:|
| num_trials     | Evaluate the integration for num_trials times. Better kept within 10. | 10 | 5 |
| depth | For importance sampling. A domain is magnified for depth times. Better kept within 3. |3|2|
| num_chunks_in_one_dimension     | The number of chunks users want to set along one dimension | 10 | 4 |
| sigma_multiplier | Only domains that have very large standardand deviations (hence, very unstable) should be magnified and re-evaluated. Domains which are beyond sigma_multiplication * &sigma; should be recalculated.|3|4|

#### Attention
The user defined function must be organized in string format as shown in the following example. And the function name in the string mutst be `fun`, something like:
```
# user defined function
fun = """ 
import math
# define a device function that should be used by cuda kernel
@cuda.jit(device=True)
def fun(x): # here the function name must be set as `fun`
    return xxx
"""
```

#### examples:
```
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

MC = MCintegral(my_func = fun, domain = [[0,10],[0,10],[0,10],[0,10],[0,10],[0,10]], head_node_address = "210.45.78.43:6789",
                depth = depth, sigma_multiplier = sigma_multiplier, num_trials = num_trials,
                num_chunks_in_one_dimension = num_chunks_in_one_dimension)

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
print(time.time()-a)

output:
total number of GPUs:  4
(pid=28535) current batch: 3, trial number: 5
(pid=28536) current batch: 1, trial number: 5
(pid=28537) current batch: 2, trial number: 5
(pid=28538) current batch: 0, trial number: 5
(pid=28535) current batch: 4, trial number: 5
(pid=28536) current batch: 5, trial number: 5
(pid=28537) current batch: 6, trial number: 5
(pid=28538) current batch: 7, trial number: 5
(pid=28536) current batch: 9, trial number: 5
(pid=28535) current batch: 8, trial number: 5
(pid=28537) current batch: 10, trial number: 5
(pid=28538) current batch: 11, trial number: 5
(pid=28536) current batch: 12, trial number: 5
(pid=28535) current batch: 13, trial number: 5
(pid=28537) current batch: 14, trial number: 5
137 hypercube(s) need(s) to be recalculated, to save time, try increasing sigma_multiplier.
result = -48.473634536766795    std = 1.9871465878803765
38.989293813705444
```

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) More Help

**One should read the [documentation](https://numba.pydata.org/numba-doc/dev/cuda/index.html) for the Numba package's CUDA capabilities when trying to use this package.** ZMCintegral is only compatible with device functions as Numba does not support dynamic parallelism. This is important when designing the integrated function.

Issues with CUDA should first be resolved by looking at the [CUDA documentation](https://docs.nvidia.com/cuda/index.html).


## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) License
----

The package is coded by Jun-Jie Zhang and checked by Hong-Zhong Wu of University of Science and Technology of China.

**This package is free**
you can redistribute it and/or modify it under the terms of 
the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
