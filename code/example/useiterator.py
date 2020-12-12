# this code is about that how to use iterator
import tensorflow as tf
import numpy as np
def main():
    example_2()
def example_1():
    dataset_train = tf.data.Dataset.range(100).batch(10)
    dataset_test = tf.data.Dataset.range(100).map(lambda x:x+10)
    dataset_train = dataset_train.map(lambda x:x+1)
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
    dataset_train.output_shapes)
    train_op = iterator.make_initializer(dataset_train)
    test_op = iterator.make_initializer(dataset_test)
    element = iterator.get_next()
    with tf.Session() as sess:
        while 1:
            try:
                sess.run(test_op)
                test = sess.run(element)
                print("test:{}".format(test))
                sess.run(train_op)
                res = sess.run(element)
                print("train:{}".format(res))
            except tf.errors.OutOfRangeError:
                break
def example_2():
    dataset_train = tf.data.Dataset.from_tensor_slices()
    dataset_test = tf.data.Dataset.range(100).map(lambda x:x+10)
    dataset_train = dataset_train.map(lambda x:x+1)

    # make a uninitialnized iterator
    handle = tf.placeholder(tf.string)
    iterator = tf.data.Iterator.from_string_handle(handle,
    dataset_train.output_types,dataset_train.output_shapes)
    element = iterator.get_next()

    train_op = dataset_train.make_initializable_iterator()
    test_op = dataset_test.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run([train_op.initializer,test_op.initializer])
        train_handle = sess.run(train_op.string_handle())
        test_handle = sess.run(test_op.string_handle())
        while 1:
            try:
                ele = sess.run(element,feed_dict={handle:train_handle})
                print("trian:",ele)
                ele = sess.run(element,feed_dict={handle:test_handle})
                print("test:",ele)
            except tf.errors.OutOfRangeError:
                break
def example_3():
    def tran(dataset):
        return dataset.map(lambda x:x+1).map(lambda x:x*x)
    train_data = tf.data.Dataset.from_tensor_slices(([1,2,3,4,5],[11,22,33,44,55])).shuffle(3).batch(3)
    iterator = tf.data.Iterator.from_structure(train_data.output_types,train_data.output_shapes)
    train_op = iterator.make_initializer(train_data.repeat(1))
    ele = iterator.get_next()
    with tf.Session() as sess:
        sess.run(train_op)
        while 1 :
            try:
                eles = sess.run(ele)
                print(eles)
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(tf.one_hot([1,2,3],10)))