## 0.1. 日志
linux平台的pycharm不能用中文输入法实在太难受了，强行锻炼我的英文写作能力

基于Cascade CNN的人脸检测算法论文复现

1. 已经实现了detection-net12、net24、net48 并且已经训练好了，但是在训练cal-net的时候，一直无法收敛，这个问题困扰了很久
## 0.2. 说明
    model.py:定义了模型
    train.py:训练了模型
    maker.py:对数据及进行处理
    classifier.py:对model进行封装，并且进行了测试
    layer.py:对卷积操作、池化操作进行了封装
    dataset.py:使用tensorflow.data.Dateset对处理好的数据集进行封装
## 效果图
![1](/assets/1.png)

