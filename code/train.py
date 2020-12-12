from data.dataset import ImageDateSet
from model import Detect_12Net
import tensorflow as tf
import numpy as np

def train():
    imgdataset = ImageDateSet('/home/dataset/FDDB/pos',
                              '/home/dataset/FDDB/neg',
                              batch=100,img_size=(48,48))
    train_op, ele_train, test_op, ele_test = imgdataset.getIterator()
    detect_12net = Detect_12Net(size=(48,48,3))

    loss = tf.summary.scalar('loss',detect_12net.loss)
    acc_v = tf.placeholder(dtype=tf.float32)
    acc_summary = tf.summary.scalar("acc",acc_v)
    loss_summary = tf.summary.merge([loss])
    step = 0
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
            _,s= sess.run([detect_12net.train_step,loss_summary], feed_dict={detect_12net.inputs: imgs,
                                                    detect_12net.targets: target})
            fw.add_summary(s, step)
            if step % 5 == 0:
                test_input, test_target, test_patterns = sess.run(ele_test)
                lossv,net12_out= sess.run([detect_12net.loss,detect_12net.fc4_out], feed_dict={detect_12net.inputs: test_input,
                                                               detect_12net.targets: test_target
                                                               })

                acc = np.sum(np.argmax(net12_out, axis=1)==np.argmax(target,axis=1)) / len(net12_out)
                acc_s= sess.run(acc_summary,feed_dict={acc_v:acc})
                fw.add_summary(acc_s,step)
                print('step {},loss:{},acc:{}'.format(step, lossv,acc))
                saver.save(sess,global_step=step,save_path='../models/', write_meta_graph=False)
            step += 1


if __name__ == '__main__':
    train()