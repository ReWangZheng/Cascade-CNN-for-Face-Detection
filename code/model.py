import tensorflow as tf
from layer import *
tenboard_dir = './tensorboard/test1/'


class Detect_12Net:
    def __init__(self,size = (12,12,3),lr = 0.001,is_train=True):
        #save the net12 input size
        self.size = size
        # input
        self.inputs = tf.placeholder(tf.float32,[None,size[0],size[1],size[2]],"12Net_input")
        # label
        self.targets = tf.placeholder(tf.float32,[None,2])

        #new scope
        with tf.variable_scope("12det"):
            # layer1 conv
            # kernel
            self.w_conv1 = weight_variable([3,3,size[2],16],'w_conv1')
            # bias
            self.w_bias1 = bias_variable([16],name='b_conv1')
            # output
            self.conv1_out = tf.nn.relu(conv2d(self.inputs,self.w_conv1,[1,1,1,1])+self.w_bias1)

            # pooling layer
            self.pool2_out = max_pool(self.conv1_out,[1,3,3,1],[1,2,2,1])

            # full connected layer
            self.fc3_w = weight_variable(shape=[int(size[0]/2 * size[1]/2* 16),16],
                                            name="fc3_w",layer_type="connected")
            self.fc3_b = bias_variable(shape=[16],name="fc3_b")

            self.fc3_flaten = flatten(self.pool2_out,"fc3_flatten")

            self.fc3_out = tf.nn.relu(tf.matmul(self.fc3_flaten,self.fc3_w)+self.fc3_b,"f3_out")

            # output layer
            self.fc4_w = weight_variable(shape=[16,2],name="fc4_w",layer_type="connected")
            self.fc4_b = bias_variable(shape=[2],name="fc4_b")
            self.fc4_out = tf.matmul(self.fc3_out,self.fc4_w) + self.fc4_b
            self.net_prob = tf.nn.softmax(logits=self.fc4_out)
            self.net_out = self.fc4_out
            self.net_out_std = tf.argmax(self.fc4_out, axis=1)

            if is_train:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc4_out, labels=self.targets))
                self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def accuracy(self):
        tar_o = tf.argmax(self.targets,axis=1)
        correct_prediction = tf.equal(self.net_out_std,tar_o)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc
class Detection_24net:
    def __init__(self,size = (24,24,3),lr = 0.001,is_train=True):
        #save the net24 input size
        self.size = size
        # input
        self.inputs = tf.placeholder(tf.float32,[None,size[0],size[1],size[2]],"12Net_input")
        self.from12 = tf.placeholder(tf.float32,[None,16])
        # label
        self.targets = tf.placeholder(tf.float32,[None,2])

        #new scope
        with tf.variable_scope("24det_"):
            # layer1 conv
            # kernel
            self.w_conv1 = weight_variable([5,5,size[2],64],'w_conv1')
            # bias
            self.w_bias1 = bias_variable([64],name='b_conv1')
            # output
            self.conv1_out = tf.nn.relu(conv2d(self.inputs,self.w_conv1,[1,1,1,1])+self.w_bias1)

            # pooling layer
            self.pool2_out = max_pool(self.conv1_out,[1,3,3,1],[1,2,2,1])

            # full connected layer
            self.fc3_w = weight_variable(shape=[int(size[0]/2 * size[1]/2* 64),128],
                                            name="fc3_w",layer_type="connected")
            self.fc3_b = bias_variable(shape=[128],name="fc3_b")

            self.fc3_flaten = flatten(self.pool2_out,"fc3_flatten")

            self.fc3_out = tf.nn.relu(tf.matmul(self.fc3_flaten,self.fc3_w)+self.fc3_b,"f3_out")

            # concat
            self.f3_concat = tf.concat([self.fc3_out,self.from12],axis=1)
            # output layer
            self.fc4_w = weight_variable(shape=[128+16,2],name="fc4_w",layer_type="connected")
            self.fc4_b = bias_variable(shape=[2],name="fc4_b")
            self.fc4_out = tf.matmul(self.f3_concat,self.fc4_w) + self.fc4_b
            self.net_prob = tf.nn.softmax(logits=self.fc4_out)
            self.net_out = self.fc4_out
            if is_train:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc4_out, labels=self.targets))
                self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def accuracy(self):
        net_o = tf.argmax(self.fc4_out,axis=1)
        tar_o = tf.argmax(self.targets,axis=1)
        correct_prediction = tf.equal(net_o,tar_o)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

class Detection_48net:
    def __init__(self,size = (48,48,3),lr = 0.001,is_train=True):
        #save the net24 input size
        self.size = size
        # input
        self.inputs = tf.placeholder(tf.float32,[None,size[0],size[1],size[2]],"12Net_input")
        self.from24 = tf.placeholder(tf.float32,[None,128+16])
        # label
        self.targets = tf.placeholder(tf.float32,[None,2])

        #new scope
        with tf.variable_scope("48det_"):
            # layer1 conv
            # kernel
            self.w_conv1 = weight_variable([5,5,size[2],64],'w_conv1')
            # bias
            self.w_bias1 = bias_variable([64],name='b_conv1')
            # output
            self.conv1_out = tf.nn.relu(conv2d(self.inputs,self.w_conv1,[1,1,1,1])+self.w_bias1)

            # pooling layer
            self.pool2_out = max_pool(self.conv1_out,[1,3,3,1],[1,2,2,1])

            #cov layer3
            self.w_conv3 = weight_variable([5,5,64,64],'w_conv3')
            self.w_bias3 = weight_variable([64], 'w_bias3')
            self.conv3_out = tf.nn.relu(conv2d(self.pool2_out,self.w_conv3,[1,1,1,1])+self.w_bias3)


            # pool layer4
            self.pool4_out = max_pool(self.conv3_out, [1, 3, 3, 1], [1, 2, 2, 1])


            # full connected layer
            self.fc5_w = weight_variable(shape=[int(size[0]/4 * size[1]/4* 64),256],
                                            name="fc5_w",layer_type="connected")
            self.fc5_b = bias_variable(shape=[256],name="fc5_b")

            self.fc5_flaten = flatten(self.pool4_out,"fc5_flatten")

            self.fc5_out = tf.nn.relu(tf.matmul(self.fc5_flaten,self.fc5_w)+self.fc5_b,"f5_out")

            # concat
            self.fc5_concat = tf.concat([self.fc5_out,self.from24],axis=1)
            # output layer
            self.fc6_w = weight_variable(shape=[128+16+256,2],name="fc6_w",layer_type="connected")
            self.fc6_b = bias_variable(shape=[2],name="fc6_b")
            self.fc4_out = tf.matmul(self.fc5_concat,self.fc6_w) + self.fc6_b
            self.net_prob = tf.nn.softmax(logits=self.fc4_out)
            self.net_out = self.fc4_out
            if is_train:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc4_out, labels=self.targets))
                self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def accuracy(self):
        net_o = tf.argmax(self.fc4_out,axis=1)
        tar_o = tf.argmax(self.targets,axis=1)
        correct_prediction = tf.equal(net_o,tar_o)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

class Calibrate_net12:
    def __init__(self,size = (24,24,3),patterns = 45,is_train = True,lr = 0.001):
        self.inputs = tf.placeholder(dtype=tf.float32,
                    shape=[None,size[0],size[1],size[2]],name="calib_net12_input")
        self.targets = tf.placeholder(dtype=tf.float32,
                    shape=[None,patterns],name="calib_net12_targets")
        with tf.variable_scope("calib_net_12"):
            # layer1 conv
            # kernel
            self.w_conv1 = weight_variable([3,3,size[2],32],'w_conv1')
            # bias
            self.w_bias1 = bias_variable([32],name='b_conv1')
            # output
            self.conv1_out = tf.nn.relu(conv2d(self.inputs,self.w_conv1,[1,1,1,1])+self.w_bias1)

            # pooling layer
            self.pool2_out = max_pool(self.conv1_out,[1,3,3,1],[1,2,2,1])

            # full connected layer
            self.fc3_w = weight_variable(shape=[int(size[0]/2 * size[1]/2* 32),256],
                                            name="fc3_w",layer_type="connected")
            self.fc3_b = bias_variable(shape=[256],name="fc3_b")

            self.fc3_flaten = flatten(self.pool2_out,"fc3_flatten")

            self.fc3_out = tf.nn.relu(tf.matmul(self.fc3_flaten,self.fc3_w)+self.fc3_b,"f3_out")

            # output layer
            self.fc4_w = weight_variable(shape=[256,45],name="fc4_w",layer_type="connected")
            self.fc4_b = bias_variable(shape=[45],name="fc4_b")
            self.fc4_out = tf.matmul(self.fc3_out,self.fc4_w) + self.fc4_b
            if is_train:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc4_out, labels=self.targets))
                self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def accuracy(self):
        net_o = tf.argmax(self.fc4_out,axis=1)
        tar_o = tf.argmax(self.targets,axis=1)
        correct_prediction = tf.equal(net_o,tar_o)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

def clib_net_12_test():
    c2 = Calibrate_net12()

def net48_test():
    Detection_48net()

if __name__ == '__main__':
    net48_test()