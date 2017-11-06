
import os
import cv2
import numpy as np
#导入python图片库
from PIL import Image
#用一个大佬的正脸预测结果初始化分类器
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

#初始化一个opencv提供的本地直方图脸部识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

#设置显示时的字体
font = cv2.FONT_HERSHEY_SIMPLEX

#检索整合并传递样本
def getSamples(path):
    #得到图片样本所在路径
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    #初始化存脸部图片样本用的数组
    faceSamples=[]
    #初始化人的id（标签）
    ids = []
    #对所有找到的文件夹进行检索
    for imagePath in imagePaths:
        #打开指定文件夹并将其中的所有图片转化成灰度图
        PIL_img = Image.open(imagePath).convert('L')
        #将图片转化为Numpy的数组
        img_numpy = np.array(PIL_img,'uint8')
        #得到照片id并将其中的人的id找出来
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #库函数找脸
        faces = detector.detectMultiScale(img_numpy)
        #将每种脸（不同人的）与与之对应的人的id相匹配
        for (x,y,w,h) in faces:
            #不断补充样本库
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            #按次序将样本（图片）对应的标签(id)添加进去
            ids.append(id)

    #返回样本和标签
    return faceSamples,ids

# 得到脸部样本集
faces,ids = getSamples('image')

#用脸部样本标签集训练模型
recognizer.train(faces, np.array(ids))

#打开电脑自带镜头
cam = cv2.VideoCapture(0)

#进行循环
while True:
    #按帧读取视频
    ret, im =cam.read()

    #将捕获的帧图片转化为灰度图
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #用分类器提取出其中图片中的脸部
    faces = detector.detectMultiScale(gray, 1.3,5)

    #对于提取的每个脸
    for(x,y,w,h) in faces:

        #用长方形标记
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)

        #用训练好的识别器进行识别，返回值为id和置信度
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id);

        #与已现存好的人的id进行比对
        if(Id[0] == 1 and Id[1] <= 56):
            Name = "Huang Qi"
        else:
            Name = "Unknown"

        #进行标注
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, Name, (x,y-40), font, 2, (255,255,255), 3)

    #在视频帧中展示
    cv2.imshow('im',im) 

    #定义q键关掉系统
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#将占得内存释放掉
cam.release()
cv2.destroyAllWindows()
