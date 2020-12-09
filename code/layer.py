import tensorflow as tf
import numpy as np
# this function is used to create a variable
def weight_variable(shape,name,layer_type='conv'):
    if layer_type=='conv':
        initial = tf.truncated_normal(shape,dtype=tf.float32,stddev=0.01)
    else:
        x = np.sqrt(6. / (np.prod(np.array(shape[:-1])) + shape[-1]))
        initial = tf.random_uniform(shape, minval=-x, maxval=x)
    return tf.Variable(initial,name=name)

#this function can be used to create a bias variable
def bias_variable(shape,name=None):
    initial = tf.constant(value=0.1,shape=shape)
    return tf.Variable(initial_value=initial,name=name)

# conv2d function
def conv2d(input_v,kernel,stride,mode='SAME'):
    return tf.nn.conv2d(input_v,kernel,stride,mode)

# maxpooling function
def max_pool(input_v,size,stride,mode="SAME"):
    return tf.nn.max_pool(input_v,size,stride,mode)

# flatten layer
def flatten(input_tensor,name):
    shape_list = input_tensor.shape.as_list()
    n,w,h,c = shape_list
    return tf.reshape(input_tensor,shape=[-1,w * h * c],name=name)