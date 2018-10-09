# ZMCintegral


ZMCintegral is a python package which uses Monte Carlo Method to do numerical integrations on Multi-GPU devices. It supports integrations with up to 11 multi-variables. It is capable of even more than 11 variables if time is not of the priori concern. 

# Supports

  - Multi-dimension integration
  - Multi-GPU device
  - Importance sampling


###### To understand how ZMCintegral works, please refer to

### Installation

To run ZMCintegral, the following packages needs to be pre-installed:
  - Tensorflow 1.10+
  - Numpy
  - Math

Installation of ZMCintegral via Anaconda (https://www.anaconda.com) is also supported.
In your specific environment, please use
```sh
$ conda install ZMCintegral
```
to install ZMC integral.

### Basic Example
Integration of the following expression:
![Image of expression 1](./examples/example01.png)
```sh
import ZMCintegral
import tensorflow as tf
# user defined function
def my_func(x):
    return tf.sin(x[0]+x[1]+x[2]+x[3])
# obtaining the result
result = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]]).MCresult
# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
```
ZMCintegral returns:
```
result = -1.0458884    std = 0.00041554452
```
### Advanced Usage
###### simontaneous evaluation
ZMCintegal supports the evaluation of several integrations simontaneously. For example, the following three:
![Image of expression 1](./examples/example02.png)
```sh
import ZMCintegral
import tensorflow as tf
# user defined function
def my_func(x):
    tf.sin(x[0]+x[1]+x[2]+x[3]),x[0]+x[1]+x[2]+x[3],x[0]*x[1]*x[2]*x[3]
# obtaining the result
result = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]]).MCresult
# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
```
ZMCintegral returns:
```
result = [-1.0458851 25.799936   2.249969 ]    std = [0.00040938 0.00066065 0.0002065 ]
```
###### tune parameters