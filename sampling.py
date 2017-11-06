
import cv2

#用一个大佬的正脸预测结果初始化分类器
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#打开电脑自带镜头
cam = cv2.VideoCapture(0)

#人的id，一人一个
id = 1

#计数器
count = 0

#对每一帧进行操作
while(True):

    #按帧读取视频
    lala,frame = cam.read()
    #图片转化为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #用分类器在图里找脸
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    #对每个脸进行存储
    for (x,y,w,h) in faces:
        #画框框脸
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        #计数器加一
        count += 1
        #存图片
        cv2.imwrite("image/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        #把框显示出来
        cv2.imshow('frame', frame)

    #设定相近刷新间隔，q键关
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    #采集的样本数量大于100时自动关闭相机
    elif count>100:
        break

#释放占用的内存
cam.release()
cv2.destroyAllWindows()
