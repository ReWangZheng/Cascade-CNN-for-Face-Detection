import numpy as np
import os
import matplotlib.pyplot as plt
import uuid
import cv2
base_dir = '/home/dataset/FDDB/'
data_dir = '/home/dataset/FDDB/FDDB-folds/'
background_dir = '/home/dataset/FDDB/back'
data_info = \
        ['FDDB-fold-05-ellipseList.txt',
         'FDDB-fold-01-ellipseList.txt',
         'FDDB-fold-04-ellipseList.txt',
         'FDDB-fold-06-ellipseList.txt',
         'FDDB-fold-02-ellipseList.txt',
         'FDDB-fold-03-ellipseList.txt',
         'FDDB-fold-09-ellipseList.txt',
         'FDDB-fold-10-ellipseList.txt',
         'FDDB-fold-07-ellipseList.txt',
         'FDDB-fold-08-ellipseList.txt']
pos_dir = '/home/dataset/FDDB/pos/'
neg_dir = '/home/dataset/FDDB/neg/'
def read_fddb_base():
    fddb_info = []
    for txt in data_info:
        file = os.path.join(data_dir,txt)
        with open(file) as f:
            mes_list = [mes.strip() for mes in f.readlines()]
            idx = 0
            while idx < len(mes_list):
                filename = os.path.join(base_dir,mes_list[idx] + '.jpg')
                idx+=1
                face_size = int(mes_list[idx])
                idx += 1
                face_box = []
                for i in range(0,face_size):
                    l,s,_,x,y,_,_ = mes_list[idx].split(' ')
                    x_min = float(x) - float(s)
                    y_min = float(y) - float(l)
                    W = float(s) * 2
                    H = float(l) * 2
                    face_box.append([x_min,y_min,W,H])
                    idx +=1
                fddb_info.append((filename,face_size,face_box))
    return fddb_info
def make_pos_data(size=1000):
    print('making pos data........')
    s_n = [0.83,0.91,1.0,1.1,1.21]
    x_n = [-0.17,0,-0.17]
    y_n = [-0.17,0,0.17]
    index = 0
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
    fddb_info= read_fddb_base()
    for img_info in fddb_info:
        uid = str(uuid.uuid4())
        filename,face_count,face_boxs = img_info
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_h,img_w = img.shape[:2]
        for box in face_boxs:
            x, y, w, h = np.array(box, dtype=int)
            for s_idx,s_i in enumerate(s_n):
                for x_ix,x_i in enumerate(x_n):
                    for y_idx,y_i in enumerate(y_n):
                        x1 = max(int(x - (x_i * img_w / s_i)),0)
                        y1 = max(int(y - (y_i * img_h / s_i)),0)
                        x2,y2 = (int(x1+w/s_i),int(y1+h/s_i))
                        face_img = img[y1:y2,x1:x2,:]
                        if face_img.shape[0]==0 or face_img.shape[1]==0:
                            continue
                        plt.imsave(os.path.join(pos_dir,
                                str(index)+'-'+uid + '_{}' +'_{}.jpg').format(1,(s_idx+1)*(x_ix+1)*(y_idx+1)),
                                face_img)
                        index+=1
                        if index > size:
                            print('{} images are made!'.format(index))
                            return
    print('{} positive images are made!\n'.format(index))


def slide_window(img,stride=[5,5],win=[12,12]):
    h,w = img.shape[:2]
    for y in range(0,h,stride[1]):
        for x in range(0,w,stride[0]):
            window = img[y:y+win[1],x:x+win[0]]
            yield window,[x/w,y/h,(x+win[0])/w,(y+win[1])/h]
def make_neg_data(size=1000):
    print('making pos data........')
    index = 0
    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)
    filelist = os.listdir(background_dir)
    for filename in filelist:
        file_path = os.path.join(background_dir,filename)
        img = plt.imread(file_path)
        windows = slide_window(img,stride=[80,80],win=[80,80])
        for win in windows:
            index+=1
            save_name = str(index)+'-'+str(uuid.uuid4())+'_0_99.jpg'
            plt.imsave(os.path.join(neg_dir,save_name),win)
    print('{} negative images are made!\n'.format(index))

def make_data(pos = True,neg = True):
    if pos:
        make_pos_data(size=10000)
    if neg:
        make_neg_data(size=5000)
if __name__ == '__main__':
    make_data()