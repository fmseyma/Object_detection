import torch
import cv2
import math
import os
import sys

args: list[int] = sys.argv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
cap = cv2.VideoCapture(sys.argv[1])
objectIndex = int(sys.argv[2])
cocoIndex = 0

if objectIndex == 0:
    cocoIndex = 9
    result = "trafik lambası"
elif objectIndex == 1:
    cocoIndex = 10
    result = "yangın musluğu"
elif objectIndex == 2:
    cocoIndex = 39
    result = "şişe"
elif objectIndex == 3:
    cocoIndex = 43
    result = "bıçak"
elif objectIndex == 4:
    cocoIndex = 63
    result = "laptop"
elif objectIndex == 5:
    cocoIndex = 65
    result = "kumanda"
elif objectIndex == 6:
    cocoIndex = 67
    result = "telefon"
elif objectIndex == 7:
    cocoIndex = 76
    result = "makas"
else:
    result = "tanımlanmayan nesne"
print(result)
n = 1
memory_center = [(0, 0)]
centerDistance = 0
centerDistanceList = [(0, 0)]
thickness = 2
timer = 0
color = (255, 0, 0)


if cap.isOpened() == 0:
    print("Error opening video  stream")
while cap.isOpened():
    ret, frame = cap.read()
    if ret == 1:
        if timer % 100 == 0:
            results = model(frame)
            cls = results.pandas().xyxy[0]['class']
            for i in range(len(cls)):
                if cls[i] == cocoIndex:
                    cnterCordin = [
                        (int(results.pandas().xyxy[0]['xmin'][i]) + int(results.pandas().xyxy[0]['xmax'][i])) / 2,
                        (int(results.pandas().xyxy[0]['ymin'][i]) + int(results.pandas().xyxy[0]['ymax'][i])) / 2]
                    radius = int(cnterCordin[1]) / 3
                    for i in range(len(memory_center)):
                        if cnterCordin[0] - 20 < memory_center[i][0] < cnterCordin[0] + 20 and cnterCordin[1] - 20 < \
                                memory_center[i][1] < cnterCordin[1] + 20:
                            for i in range(len(cls)):
                                if cls[i] == 0:
                                    cnterCordinPerson = [(int(results.pandas().xyxy[0]['xmin'][i]) + int(
                                        results.pandas().xyxy[0]['xmax'][i])) / 2, (
                                                                 int(results.pandas().xyxy[0]['ymin'][i]) + int(
                                                             results.pandas().xyxy[0]['ymax'][i])) / 2]
                                    distance = math.sqrt((abs(cnterCordin[0] - cnterCordinPerson[0]) ** 2) + (
                                            abs(cnterCordin[1] - cnterCordinPerson[1]) ** 2))
                                    if int(distance) > int(radius):
                                        centerDistance = math.dist(memory_center[-2], memory_center[-1])
                                        centerDistanceList.append(centerDistance)

                                        if centerDistance >= 3.17:
                                            if centerDistanceList[-2] != centerDistanceList[-1]:
                                                frame = cv2.circle(frame, (int(cnterCordin[0]), int(cnterCordin[1])),
                                                                   int(radius), color, thickness)

                                                print(str(result) + " algılandı")
                                                cv2.imwrite(f'../obj-det/images/image_{n}.png', frame)
                                                n = n + 1

                                                '''
                                                print(str(results) + " nesne algılandı")
                                                image 1/1: 1296x2304 1 person, 1 backpack, 1 bottle, 1 cup, 3 chairs, 
                                                           2 tvs, 4 laptops, 2 keyboards
                                                Speed: 4.9ms pre-process, 1885.3ms inference, 2.0ms NMS per image at 
                                                       shape (1, 3, 384, 640)
                                                '''

                    timer = 0
                    memory_center.append(cnterCordin)

        timer += 1
        cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
