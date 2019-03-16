# ZMCintegral

ZMCintegral (Numba backened) is an easy to use python package which uses Monte Carlo Evaluation Method to do numerical integrations on Multi-GPU devices. 
It supports integrations with up to 16 multi-variables, and it is capable of even more than 16 variables if time is not of the priori concern. 
ZMCintegral usually takes a few minutes to finish the task.

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Newest Features

  - Full flexibility of user defined functions
  - Multi-dimension integration
  - Multi-GPU supports
  - Stratified sampling
  - Heuristic tree search
  

> **To understand how ZMCintegral works, please refer to https://arxiv.org/pdf/1902.07916.pdf**


## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Installation

To run ZMCintegral, the following packages needs to be pre-installed:
  - Numba
  - Numpy
  - Math

Installation of ZMCintegral via Anaconda (https://anaconda.org/zhang-junjie/zmcintegral) is also supported.

In your specific environment, please use

```sh
$ conda install -c zhang-junjie zmcintegral=3.0
```
to install ZMC integral, and make sure you have Numba CUDA installed.

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Basic Example
Integration of the following expression:
![Image of expression 1](./examples/example01.png)

```sh
import math
from numba import cuda
from ZMCintegral import ZMCintegral

# user defined function
@cuda.jit(device=True)
def my_func(x):
    return math.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]])

MC.depth = 2
MC.sigma_multiplication = 5
MC.num_trials = 5

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
```
ZMCintegral returns:

```sh
result = -1.0458884    std = 0.00041554452
```

#### - tuning parameters

The following four parameters can be tuned to fit special cases.

| parameter        | usage           | example           | default  |
|:-------------:|:-------------:|:-------------:|:-----:|
| available_GPU    | Specify gpu used in calculation. | [0,1] | ALL GPUs detected |
| num_trials     | Evaluate the integration for num_trials times. Better kept within 10. | 10 | 5 |
| depth | For importance sampling. A domain is magnified for depth times. Better kept within 3. |3|2| 
| sigma_multiplication | Only domains that have very large standardand deviations (hence, very unstable) should be magnified and re-evaluated. Domains which are beyond sigma_multiplication * &sigma; should be recalculated.|3|4|

eg:

```sh
ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]],
available_GPU=[0,1],num_trials=3,depth=3,sigma_multiplication=3).evaluate()
```

#### - sampling points reconfiguration

ZMCintegral configures the sampling points automatically, 
but it also provides user-reconfigure of sampling points, eg:

```sh
import math
from numba import cuda
from ZMCintegral import ZMCintegral

# user defined function
@cuda.jit(device=True)
def my_func(x):
    return math.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]])

#############################################################################################
# sampling points reconfiguration
# total sampling points is equal to (chunk_size_x*chunk_size_multiplier)**dim, which is huge.
MC.chunk_size_x = 20
MC.chunk_size_multiplier = 3
#############################################################################################

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
```

#### - Tip: when to change chunk_size_x and chunk_size_multiplier?

If user want more points to be sampled, he/she can increase chunk_size_x and chunk_size_multiplier. **chunk_size_x * chunk_size_multiplier** equals the number of points in each dimension.


## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) License
----

The package is coded by ZHANG Junjie and checked by WU Hongzhong of University of Science and Technology of China.

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
