import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/cxk.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)  # draw=True时显示所有点

    if len(lmList) != 0:
        # print(lmList)  # 打印所有pose点的坐标变化
        print(lmList[14])  # 打印某个pose点的坐标变化
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv2.FILLED)  # 追踪14号点(右手肘）

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)  # 引入模型前取1和2会报错