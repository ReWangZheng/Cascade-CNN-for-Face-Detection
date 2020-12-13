from data.dataset import ImageDateSet
from model import Detect_12Net
import tensorflow as tf
import numpy as np
def clean():
    import os
    os.system('rm -r ../logs/* ../models/*')

def train():
    imgdataset = ImageDateSet('/home/dataset/FDDB/pos',
                              '/home/dataset/FDDB/neg',
                              batch=500)
    train_op, ele_train, test_op, ele_test = imgdataset.getIterator()
    detect_12net = Detect_12Net(lr=0.001)
    step = 0
    '''
    create summary
    '''
    with tf.name_scope('net12'):
        tf.summary.scalar('loss_value',detect_12net.loss)
        tf.summary.histogram('activation_',detect_12net.fc3_out)
        tf.summary.scalar('accuracy',detect_12net.accuracy())
    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        fw = tf.summary.FileWriter('../logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(train_op)
        sess.run(test_op)
        saver = tf.train.Saver(max_to_keep=5)
        check_model = tf.train.latest_checkpoint('../models')
        if check_model is not None:
            saver.restore(sess,check_model)
            step = int(check_model.split('-')[1])
        while 1:
            if step == 30000:
                break
            imgs, target, patterns = sess.run(ele_train)
            _,sum = sess.run([detect_12net.train_step,summary], feed_dict={detect_12net.inputs: imgs,
                                                    detect_12net.targets: target})
            fw.add_summary(sum,step)
            if step % 5 == 0:
                test_input, test_target, test_patterns = sess.run(ele_test)

                lossv,acc= sess.run([detect_12net.loss,detect_12net.accuracy()],
                                          feed_dict={detect_12net.inputs: test_input,
                                                    detect_12net.targets: test_target})
                print('step {},loss:{},acc:{}'.format(step, lossv,acc))
                saver.save(sess,global_step=step,save_path='../models/')
                print('\n')
            step += 1
if __name__ == '__main__':
    train()