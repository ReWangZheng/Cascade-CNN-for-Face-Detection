import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tensorflow.data import Dataset,Iterator
import numpy as np
import cv2
class ImageDateSet:
    def __init__(self,positive_dir,negative_dir,img_size=(12,12),batch=50):
        self.img_size = img_size
        self.batch = batch
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        file_list = os.listdir(positive_dir)
        file_list.extend(os.listdir(negative_dir))
        data_label_info = [(file_name.split('_')[1],file_name.split('_')[2][:-4]) for file_name in file_list]
        self.data_info = []
        for idx,filename in enumerate(file_list):
            lab,pattern = data_label_info[idx]
            if lab == '0':
                self.data_info.append((os.path.join(negative_dir,filename),lab,pattern))
            elif lab== '1':
                self.data_info.append((os.path.join(positive_dir,filename),lab,pattern))
        self.data_train,self.data_test = train_test_split(self.data_info)

        self.dataset_train = Dataset.from_generator(self.generator,(tf.int32,tf.int32,tf.int32)
                                                    ,args=[self.data_train])
        self.dataset_train = self.dataset_train.shuffle(500).batch(self.batch).repeat()


        self.dataset_test = Dataset.from_generator(self.generator,(tf.int32,tf.int32,tf.int32)
                                                    ,args=[self.data_test])
        self.dataset_test = self.dataset_test.shuffle(500).batch(self.batch).repeat()
    def getIterator(self):
        iter_train = self.dataset_train.make_initializable_iterator()
        train_op = iter_train.make_initializer(self.dataset_train)
        element_train = iter_train.get_next()

        iter_test = self.dataset_test.make_initializable_iterator()
        test_op = iter_test.make_initializer(self.dataset_test)
        element_test = iter_test.get_next()
        return train_op,element_train,test_op,element_test
    def generator(self,info_sets):
        for info in info_sets:
            fp,label,cls = info
            img = plt.imread(fp.decode('UTF-8'))
            img = cv2.resize(img,dsize=self.img_size)
            label_hot = np.zeros(shape=[2],dtype=int)
            label_hot[int(label)] = 1
            cls_hot = np.zeros(shape=[45], dtype=int)
            if cls!=b'99':
                cls_hot[int(cls)-1] = 1
            yield img,label_hot,cls_hot
    def load_img(self,img_path):
        print(img_path[0])
        return img_path

# Dataset test
if __name__ == '__main__':
    imgdataset = ImageDateSet('/home/dataset/FDDB/pos','/home/dataset/FDDB/neg')
    train_op, ele_train, test_op, ele_test = imgdataset.getIterator()
    with tf.Session() as sess:
        sess.run(train_op)
        sess.run(test_op)
        imgs,target,patterns=sess.run(ele_train)
        print(imgs.shape)
        print(target)
        print(patterns)