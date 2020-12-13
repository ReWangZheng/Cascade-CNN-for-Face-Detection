import tensorflow as tf
import numpy as np
import cv2
from data.dataset import one_hot
from model import Detect_12Net
import matplotlib.pyplot as plt
import os
from data.dataset import show_img
from data.dataset import img_pyramids
from data.dataset import drwa_bbox,NMS
class Classifier:
    def __init__(self,model_path):
        self.net12 = Detect_12Net()
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess,model_path)
    def score(self,x_test,y_target,is_one_hot = None):
        if is_one_hot is not None:
            y_target = one_hot(y_target,is_one_hot)
        feed_data = {self.net12.inputs:x_test
                    ,self.net12.targets:y_target}

        acc,prob = self.sess.run([self.net12.accuracy(),self.net12.net_prob],
                        feed_dict=feed_data)
        return acc, prob
    def predict_one(self,image):
        h,w,_= image.shape
        pyrs,wins,bboxs= img_pyramids(image,pyramcount=1)
        x_input = np.empty(shape=[len(wins),12,12,3])
        for idx,win in enumerate(wins):
            x_input[idx] = cv2.resize(win,dsize=(12,12))
        prob = self.sess.run([self.net12.net_prob],feed_dict={self.net12.inputs:x_input})[0]
        pred = np.argmax(prob>0.9,axis=1)
        print(np.sum(pred))
        boxs = []
        probs = []
        for idx,i in enumerate(pred):
            if i==0:
                continue
            x1_r,y1_r,x2_r,y2_r = bboxs[idx]
            x1,y1,x2,y2 = int(x1_r * w),int(y1_r * h),int(x2_r * w), int(y2_r * h)
            boxs.append([x1,y1,x2,y2])
            probs.append(prob[idx][i])
        res_box = NMS(boxs,probs,convert=False,thread=0.2)
        drwa_bbox(image,res_box,convert=False,show=True)
def test_classsifer():
    base_dir = '/home/dataset/FDDB/neg/'
    file_list = os.listdir(base_dir)
    x_test = np.empty(shape=[500,12,12,3],dtype=int)
    for idx,filename in enumerate(file_list):
        file = os.path.join(base_dir,filename)
        img = plt.imread(file)
        new_img = cv2.resize(img,dsize=(12,12))
        if idx > 499:
            break
        x_test[idx] = new_img
    print(x_test.shape)
    y_test = np.array([0 for i in range(500)])
    print(y_test.shape)
    face_classifer = Classifier(model_path=tf.train.latest_checkpoint('../models/'))
    score,pred = face_classifer.score(x_test,y_test,is_one_hot=2)
    print(score,pred)
def test_predict():
    img = plt.imread('example/img_1275.jpg')
    show_img(img)
    face_classifer = Classifier(model_path=tf.train.latest_checkpoint('../models/'))
    face_classifer.predict_one(img)

if __name__ == '__main__':
    test_predict()