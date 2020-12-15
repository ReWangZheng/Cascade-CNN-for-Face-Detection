import tensorflow as tf
import numpy as np
import cv2
from data.dataset import one_hot
from model import Detect_12Net, Calibrate_net12, Detection_24net,Detection_48net
import matplotlib.pyplot as plt
import os
from data.dataset import show_img
from data.dataset import img_pyramids
from data.dataset import drwa_bbox, NMS


class Classifier_Det12:
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net12 = Detect_12Net()
            self.sess = tf.Session()
            tf.train.Saver(self.graph.get_collection(tf.GraphKeys.VARIABLES)).restore(self.sess, model_path)

    def score(self, x_test, y_target, is_one_hot=None):
        with self.graph.as_default():
            if is_one_hot is not None:
                y_target = one_hot(y_target, is_one_hot)
            feed_data = {self.net12.inputs: x_test
                , self.net12.targets: y_target}

            acc, prob = self.sess.run([self.net12.accuracy(), self.net12.net_prob],
                                      feed_dict=feed_data)
            return acc, prob

    def predict_one(self, image):
        with self.graph.as_default():
            h, w, _ = image.shape
            pyrs, wins, bboxs = img_pyramids(image, pyramcount=3)
            x_input = np.empty(shape=[len(wins), 12, 12, 3])
            for idx, win in enumerate(wins):
                x_input[idx] = cv2.resize(win, dsize=(12, 12))
            prob = self.sess.run([self.net12.net_prob], feed_dict={self.net12.inputs: x_input})[0]
            pred = np.argmax(prob > 0.9, axis=1)
            print(np.sum(pred))
            boxs = []
            probs = []
            for idx, i in enumerate(pred):
                if i == 0:
                    continue
                x1_r, y1_r, x2_r, y2_r = bboxs[idx]
                x1, y1, x2, y2 = int(x1_r * w), int(y1_r * h), int(x2_r * w), int(y2_r * h)
                boxs.append([x1, y1, x2, y2])
                probs.append(prob[idx][i])
            res_box = NMS(boxs, probs, convert=False, thread=0.2)
            drwa_bbox(image, res_box, convert=False, show=True)

    def predict(self, data, resize=False):
        data_input = np.empty(shape=[len(data), 12, 12, 3])
        if resize:
            for idx in range(0, len(data)):
                img = data[idx]
                new_img = cv2.resize(np.array(img, dtype='uint8'), dsize=(12, 12))
                data_input[idx] = new_img
        else:
            data_input = data
        with self.graph.as_default():
            value = self.sess.run(self.net12.net_out_std,
                                  feed_dict={self.net12.inputs: data_input})
            return value

    def face_prob(self, data, resize=False):
        data_input = np.empty(shape=[len(data), 12, 12, 3])
        if resize:
            for idx in range(0, len(data)):
                img = data[idx]
                new_img = cv2.resize(np.array(img, dtype='uint8'), dsize=(12, 12))
                data_input[idx] = new_img
        else:
            data_input = data
        with self.graph.as_default():
            value = self.sess.run(self.net12.net_prob,
                                  feed_dict={self.net12.inputs: data_input})
            return value[:, 1]

    def get_f12(self, data, resize=False):
        data_input = np.empty(shape=[len(data), 12, 12, 3])
        if resize:
            for idx in range(0, len(data)):
                img = data[idx]
                new_img = cv2.resize(np.array(img, dtype='uint8'), dsize=(12, 12))
                data_input[idx] = new_img
        else:
            data_input = data
        with self.graph.as_default():
            return self.sess.run(self.net12.fc3_out, feed_dict={self.net12.inputs: data_input})


class Classifier_cascadeNet_12_24:
    def __init__(self, model_path):
        self.net12 = Classifier_Det12(tf.train.latest_checkpoint('../models/net12'))
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net24 = Detection_24net()
            self.sess = tf.Session()
            tf.train.Saver(self.graph.get_collection(tf.GraphKeys.VARIABLES)).restore(self.sess, model_path)
    def predict(self, data):
        pred_net12 = self.net12.predict(data, resize=True)
        face_net12_index = [pred_net12 == 1]
        data_pred_casd_24 = data[tuple(face_net12_index)]
        data_pred_from_12 = self.net12.get_f12(data_pred_casd_24, resize=True)
        with self.graph.as_default():
            pred = self.sess.run(self.net24.net_prob,
                                 feed_dict={self.net24.inputs: data_pred_casd_24, self.net24.from12: data_pred_from_12})
            casd_24_pred = np.argmax(pred, axis=1)
        pred_net12[tuple(face_net12_index)] = casd_24_pred
        return pred_net12

    def predict_one(self, image):
        # get the input image size
        h, w, _ = image.shape
        # Image pyramids process images
        pyrs, wins, bboxs = img_pyramids(image, pyramcount=4)
        bboxs = np.array(bboxs)
        # create empty temp
        x_input = np.empty(shape=[len(wins), 24, 24, 3])
        # resize the image
        for idx, win in enumerate(wins):
            x_input[idx] = cv2.resize(win, dsize=(24, 24))
        prob = self.net12.face_prob(data=x_input, resize=True)
        face_idex = np.argwhere(prob > 0.8).flatten()

        print("thare are {} face after cascade-1".format(len(face_idex)))

        # get all the face that pass cascade-1
        net24_input = x_input[face_idex]
        bboxs = bboxs[face_idex]

        face_prob_net24 = self.face_prob(net24_input)
        # get the face index
        net24_face_index = np.argwhere(face_prob_net24 > 0.8).flatten()
        bboxs = bboxs[net24_face_index]
        print("thare are {} face after cascade-2".format(len(bboxs)))
        boxs = []
        probs = []
        for idx in range(0,len(bboxs)):
            x1_r, y1_r, x2_r, y2_r = bboxs[idx]
            x1, y1, x2, y2 = int(x1_r * w), int(y1_r * h), int(x2_r * w), int(y2_r * h)
            boxs.append([x1, y1, x2, y2])
            probs.append(prob[idx])
        res_box = NMS(boxs, probs, convert=False, thread=0.1)
        return drwa_bbox(image, res_box, convert=False, show=False)

    def score(self, x_test, y_target, is_one_hot=None):
        pred = self.predict(x_test)
        return np.mean(pred == y_target)
    def face_prob(self,data,resize=False):
        data_input = np.empty(shape=[len(data), 24, 24, 3])
        if resize:
            for idx in range(0, len(data)):
                img = data[idx]
                new_img = cv2.resize(np.array(img, dtype='uint8'), dsize=(24, 24))
                data_input[idx] = new_img
        else:
            data_input = data
        # start to compute
        with self.graph.as_default():
            # get the full-connected layer outout of net12
            net12_fc = self.net12.get_f12(data_input, resize=True)
            # start to compute
            with self.graph.as_default():
                face_prob_net24 = self.sess.run(self.net24.net_prob,
                                                feed_dict={self.net24.inputs: data_input,
                                                           self.net24.from12: net12_fc})[:, 1]
            return face_prob_net24
    def from_fc24(self,data,resize = False):
        input_data = data
        if resize:
            input_data = np.empty([len(data),24,24,3])
            for idx in range(0, len(data)):
                img = data[idx]
                new_img = cv2.resize(np.array(img, dtype='uint8'), dsize=(24, 24))
                input_data[idx] = new_img
        with self.graph.as_default():
            val = self.sess.run(self.net24.f3_concat,feed_dict={self.net24.inputs:input_data,
                                                                self.net24.from12:self.net12.get_f12(data,True)})
        return val

class Classifier_cascadeNet_12_24_48:
    def __init__(self, model_path):
        self.net24 = Classifier_cascadeNet_12_24(tf.train.latest_checkpoint('../models/net24'))
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.net48 = Detection_48net()
            self.sess = tf.Session()
            tf.train.Saver(self.graph.get_collection(tf.GraphKeys.VARIABLES)).restore(self.sess, model_path)
    def predict(self, data):
        with self.graph.as_default():
            val = self.sess.run(self.net48.net_out,feed_dict={self.net48.inputs:data,self.net48.from24:self.net24.from_fc24(data,resize=True)})
        return val
    def predict_one(self, image):
        # get the input image size
        h, w, _ = image.shape
        # Image pyramids process images
        pyrs, wins, bboxs = img_pyramids(image, pyramcount=4)
        # create empty temp
        x_input = np.empty(shape=[len(wins), 48, 48, 3])
        # resize the image
        for idx, win in enumerate(wins):
            x_input[idx] = cv2.resize(win, dsize=(48, 48))

        face_12_prob = self.net24.net12.face_prob(x_input,resize=True)
        index = np.argwhere(face_12_prob>0.8).flatten()
        bboxs = np.array(bboxs)
        x_input = x_input[index]
        bboxs = bboxs[index]
        face_24_prob = self.net24.face_prob(data=x_input, resize=True)
        index = np.argwhere(face_24_prob>0.8).flatten()
        x_input = x_input[index]
        bboxs = bboxs[index]
        face_48_prob = self.face_prob(x_input)
        index = np.argwhere(face_48_prob>0.8).flatten()
        bboxs = bboxs[index]
        print("thare are {} face after cascade-2".format(len(bboxs)))
        boxs = []
        probs = []
        for idx in range(0,len(bboxs)):
            x1_r, y1_r, x2_r, y2_r = bboxs[idx]
            x1, y1, x2, y2 = int(x1_r * w), int(y1_r * h), int(x2_r * w), int(y2_r * h)
            boxs.append([x1, y1, x2, y2])
            probs.append(face_48_prob[idx])
        res_box = NMS(boxs, probs, convert=False, thread=0.1)
        return drwa_bbox(image, res_box, convert=False, show=False)

    def score(self, x_test, y_target, is_one_hot=None):
        pred = self.face_prob(x_test)
        face = np.zeros([len(pred),])
        face[pred>0.5]=1
        return np.mean(face == y_target)
    def face_prob(self,data):
        # start to compute
        with self.graph.as_default():
            # get the full-connected layer outout of net12
            net24_fc = self.net24.from_fc24(data, resize=True)
            # start to compute
            with self.graph.as_default():
                face_prob_net24 = self.sess.run(self.net48.net_prob,
                                                feed_dict={self.net48.inputs: data,
                                                           self.net48.from24: net24_fc})[:, 1]
            return face_prob_net24

def test_classsifer():
    base_dir = '/home/dataset/FDDB/face_pos/'
    file_list = os.listdir(base_dir)
    x_test = np.empty(shape=[500, 12, 12, 3], dtype=int)
    for idx, filename in enumerate(file_list):
        file = os.path.join(base_dir, filename)
        img = plt.imread(file)
        new_img = cv2.resize(img, dsize=(12, 12))
        if idx > 499:
            break
        x_test[idx] = new_img
    print(x_test.shape)
    y_test = np.array([1 for i in range(500)])
    print(y_test.shape)
    face_classifer = Classifier_Det12(model_path=tf.train.latest_checkpoint('../models/net12'))
    score, pred = face_classifer.score(x_test, y_test, is_one_hot=2)
    print(score, pred)


def test_predict():
    img = plt.imread('example/img_14006.jpg')
    show_img(img)
    face_classifer = Classifier_Det12(model_path=tf.train.latest_checkpoint('../models/net12/'))
    face_classifer.predict_one(img)
def test_classsifer12_24():
    base_dir = '/home/dataset/FDDB/face_pos/'
    file_list = os.listdir(base_dir)
    x_test = np.empty(shape=[500, 24, 24, 3], dtype=int)
    for idx, filename in enumerate(file_list):
        file = os.path.join(base_dir, filename)
        img = plt.imread(file)
        new_img = cv2.resize(img, dsize=(24, 24))
        if idx > 499:
            break
        x_test[idx] = new_img
    print(x_test.shape)
    y_test = np.array([1 for i in range(500)])
    print(y_test.shape)
    face_classifer = Classifier_cascadeNet_12_24(model_path=tf.train.latest_checkpoint('../models'))
    pred = face_classifer.score(x_test, y_test, is_one_hot=2)
    print(pred)

def test_classsifer_12_24_48():
    base_dir = '/home/dataset/FDDB/face_pos/'
    file_list = os.listdir(base_dir)
    x_test = np.empty(shape=[500, 48, 48, 3], dtype=int)
    for idx, filename in enumerate(file_list):
        file = os.path.join(base_dir, filename)
        img = plt.imread(file)
        new_img = cv2.resize(img, dsize=(48, 48))
        if idx > 499:
            break
        x_test[idx] = new_img
    print(x_test.shape)
    y_test = np.array([1 for i in range(500)])
    print(y_test.shape)
    face_classifer = Classifier_cascadeNet_12_24_48(model_path=tf.train.latest_checkpoint('../models/net48'))
    score = face_classifer.score(x_test, y_test, is_one_hot=2)
    print(score)


def test_predict_24_12():
    img = plt.imread('example/img_14006.jpg')
    # 获取摄像头
    capture = cv2.VideoCapture(1)
    capture.set(3, 480)
    face_classifer = Classifier_cascadeNet_12_24(model_path=tf.train.latest_checkpoint('../models/net24'))
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, image = capture.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = face_classifer.predict_one(img)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("image", image)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("test.jpg", image)
        elif k == ord("q"):
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
def test_predict_48_24_12():
    img = plt.imread('example/img_1275.jpg')
    # 获取摄像头
    capture = cv2.VideoCapture(1)
    capture.set(3, 480)
    face_classifer = Classifier_cascadeNet_12_24_48(model_path=tf.train.latest_checkpoint('../models/net48'))
    img = face_classifer.predict_one(img)
    show_img(img)
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, image = capture.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = face_classifer.predict_one(img)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("image", image)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("test.jpg", image)
        elif k == ord("q"):
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_predict_48_24_12()
