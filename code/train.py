from data.dataset import ImageDateSet
from model import Detect_12Net, Calibrate_net12, Detection_24net,Detection_48net
import tensorflow as tf
import numpy as np
import os
from Classifier import Classifier_Det12,Classifier_cascadeNet_12_24

def clean():
    import os
    os.system('rm -r ../logs/* ../models/*')


def train_net12():
    detection_dateset = ImageDateSet('/home/dataset/FDDB/face_pos',
                                     '/home/dataset/FDDB/neg',
                                     batch=800)
    det_train_op, det_ele_train, det_test_op, det_ele_test = detection_dateset.getIterator()
    detect_12net = Detect_12Net(lr=0.0001)
    step = 0
    '''
    create summary
    '''
    with tf.name_scope('detation_net12'):
        tf.summary.scalar('loss_value', detect_12net.loss)
        tf.summary.scalar('accuracy', detect_12net.accuracy())
    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        fw = tf.summary.FileWriter('../logs/', sess.graph)

        # in order to get the device's message
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()

        # init the variable
        sess.run(tf.global_variables_initializer())
        sess.run([det_train_op, det_test_op])
        saver = tf.train.Saver(max_to_keep=5)
        check_model = tf.train.latest_checkpoint('../models/net12')
        if check_model is not None:
            saver.restore(sess, check_model)
            step = int(check_model.split('-')[1])
        while 1:
            if step == 30000:
                break

            imgs_det, target_det, patterns_det = sess.run(det_ele_train)

            # train detection net
            sess.run([detect_12net.train_step],
                     feed_dict={detect_12net.inputs: imgs_det,
                                detect_12net.targets: target_det})
            # summary
            if step % 5 == 0:
                test_input, test_target, test_patterns = sess.run(det_ele_test)
                lossv1, acc1, summ = sess.run([detect_12net.loss, detect_12net.accuracy(),
                                                               summary],
                                                              feed_dict={detect_12net.inputs: test_input,
                                                                         detect_12net.targets: test_target,
                                                                         })
                fw.add_summary(summ, step)
                print('step {},det loss:{},acc:{}; '.format(step, lossv1, acc1))
                saver.save(sess, global_step=step, save_path='../models/')
            if step % 15 == 0:
                sess.run([detect_12net.loss, detect_12net.accuracy()],
                         feed_dict={detect_12net.inputs: test_input, detect_12net.targets: test_target},
                         options=run_options, run_metadata=metadata)
                fw.add_run_metadata(run_metadata=metadata, tag='net12_global{}'.format(step), global_step=step)
            step += 1




def train_net24():
    detect_12 = Classifier_Det12(tf.train.latest_checkpoint('../models/net12/'))
    detection_dateset = ImageDateSet('/home/dataset/FDDB/face_pos',
                                     '/home/dataset/FDDB/neg',
                                     batch=800,
                                     img_size=(24, 24)
                                     )
    det_train_op, det_ele_train, det_test_op, det_ele_test = detection_dateset.getIterator()
    detect_24net = Detection_24net(lr=0.0001)
    step = 0
    '''
    create summary
    '''
    with tf.name_scope('detation_net24'):
        tf.summary.scalar('loss_value', detect_24net.loss)
        tf.summary.scalar('accuracy', detect_24net.accuracy())
    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        fw = tf.summary.FileWriter('../logs/', sess.graph)

        # in order to get the device's message
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()

        # init the variable
        sess.run(tf.global_variables_initializer())
        sess.run([det_train_op, det_test_op])
        saver = tf.train.Saver(max_to_keep=5)

        check_model = tf.train.latest_checkpoint('../models')
        if check_model is not None:
            saver.restore(sess, check_model)
            step = int(check_model.split('-')[1])
        while 1:
            if step == 30000:
                break

            imgs_det, target_det, patterns_det = sess.run(det_ele_train)
            # train detection net
            net_f12 = detect_12.get_f12(imgs_det,resize=True)
            sess.run([detect_24net.train_step],
                     feed_dict={detect_24net.inputs: imgs_det,
                                detect_24net.targets: target_det,
                                detect_24net.from12:net_f12
                                }
                     )

            # summary
            if step % 5 == 0:
                test_input, test_target, test_patterns = sess.run(det_ele_test)
                lossv, acc2, summ = sess.run([detect_24net.loss, detect_24net.accuracy(),
                                                               summary],
                                                              feed_dict={detect_24net.inputs: test_input,
                                                                         detect_24net.targets: test_target,
                                                                         detect_24net.from12:detect_12.get_f12(test_input,resize=True)
                                                                         })
                fw.add_summary(summ, step)
                print('step {},det loss:{},acc:{}'.format(step, lossv, acc2))
                saver.save(sess, global_step=step, save_path='../models/')
            if step % 15 == 0:
                sess.run([detect_24net.loss, detect_24net.accuracy()],
                         feed_dict={detect_24net.inputs: test_input,
                                    detect_24net.targets: test_target,
                                    detect_24net.from12:detect_12.get_f12(test_input,resize=True)
                                    },
                         options=run_options, run_metadata=metadata)
                fw.add_run_metadata(run_metadata=metadata, tag='net24_global{}'.format(step), global_step=step)
            step += 1

def train_net48():
    detection_dateset = ImageDateSet('/home/dataset/FDDB/face_pos',
                                     '/home/dataset/FDDB/neg',
                                     batch=800,img_size=(48,48))
    det_train_op, det_ele_train, det_test_op, det_ele_test = detection_dateset.getIterator()
    detect_48net = Detection_48net(lr=0.001)
    det_24_12 = Classifier_cascadeNet_12_24(tf.train.latest_checkpoint('../models/net24'))
    step = 0
    '''
    create summary
    '''
    with tf.name_scope('detation_net48'):
        tf.summary.scalar('loss_value', detect_48net.loss)
        tf.summary.scalar('accuracy', detect_48net.accuracy())
    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        fw = tf.summary.FileWriter('../logs/', sess.graph)

        # in order to get the device's message
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()

        # init the variable
        sess.run(tf.global_variables_initializer())
        sess.run([det_train_op, det_test_op])
        saver = tf.train.Saver(max_to_keep=5)
        check_model = tf.train.latest_checkpoint('../models/net48')
        if check_model is not None:
            saver.restore(sess, check_model)
            step = int(check_model.split('-')[1])
        while 1:
            if step == 30000:
                break

            imgs_det, target_det, patterns_det = sess.run(det_ele_train)

            # train detection net
            sess.run([detect_48net.train_step],
                     feed_dict={detect_48net.inputs: imgs_det,
                                detect_48net.targets: target_det,
                                detect_48net.from24:det_24_12.from_fc24(imgs_det,True)
                                })
            # summary
            if step % 5 == 0:
                test_input, test_target, test_patterns = sess.run(det_ele_test)
                lossv1, acc1, summ = sess.run([detect_48net.loss, detect_48net.accuracy(),
                                                               summary],
                                                              feed_dict={detect_48net.inputs: test_input,
                                                                         detect_48net.targets: test_target,
                                                                         detect_48net.from24:det_24_12.from_fc24(test_input,True)
                                                                         })
                fw.add_summary(summ, step)
                print('step {},det loss:{},acc:{}; '.format(step, lossv1, acc1))
                saver.save(sess, global_step=step, save_path='../models/net48')
            step += 1

def train_calibrate():
    calibrate_dataset = ImageDateSet('/home/dataset/FDDB/pos',
                                     None,
                                    img_size=(24,24),
                                     batch=150)
    cal_train_op, cal_ele_train, cal_test_op, cal_ele_test = calibrate_dataset.getIterator()
    calibrate_12net = Calibrate_net12(lr=0.001)
    step = 0
    '''
    create summary
    '''
    with tf.name_scope("calibrate_net12"):
        tf.summary.scalar("loss_value", calibrate_12net.loss)
        tf.summary.scalar("accuracy", calibrate_12net.accuracy())

    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        fw = tf.summary.FileWriter('../logs/', sess.graph)
        # in order to get the device's message
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()

        # init the variable
        sess.run(tf.global_variables_initializer())
        sess.run([cal_train_op, cal_test_op])
        saver = tf.train.Saver(max_to_keep=5)
        check_model = tf.train.latest_checkpoint('../models/cal12')
        if check_model is not None:
            saver.restore(sess, check_model)
            step = int(check_model.split('-')[1])
        while 1:
            if step == 30000:
                break
            imgs, target, patterns = sess.run(cal_ele_train)
            # train calibrate net
            sess.run([calibrate_12net.train_step],
                     feed_dict={calibrate_12net.inputs: imgs,
                                calibrate_12net.targets: patterns})
            # summary
            if step % 5 == 0:
                cal__test_input, cal_test_target, cal_patterns = sess.run(cal_ele_test)
                cal_loss, acc2, summ = sess.run([calibrate_12net.loss, calibrate_12net.accuracy(),
                                                 summary],
                                                feed_dict={calibrate_12net.inputs: cal__test_input,
                                                           calibrate_12net.targets: cal_patterns
                                                           })
                fw.add_summary(summ, step)
                print('step {}, cal loss{},acc{}'.format(step, cal_loss, acc2))
                saver.save(sess, global_step=step, save_path='../models/cal12/')
            step += 1

if __name__ == '__main__':
    train_calibrate()
