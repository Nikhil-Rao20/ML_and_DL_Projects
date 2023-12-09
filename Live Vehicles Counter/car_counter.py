from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


def write_text(img, x1, y1, x2, y2, classname, conf, color=(255,0,255)):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Adjust the font scale as needed
    font_thickness = 1
    text_size, _ = cv2.getTextSize(f"{classname} {conf}", font, font_scale, font_thickness)
    text_width, text_height = text_size
    bg_color = color
    bg_position = (x1, y1 - text_height - 2)  # Adjust the vertical offset as needed
    bg_size = (text_width + 5, text_height + 2)
    cv2.rectangle(img, bg_position, (bg_position[0] + bg_size[0], bg_position[1] + bg_size[1]), bg_color, -1)
    text_position = (x1 + 2, y1 - 2)  # Adjust the vertical and horizontal offset as needed
    cv2.putText(img, f"{classname} {conf}", text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


cap = cv2.VideoCapture("Vehicles_Video.mp4") #This code is for a video based classification

model = YOLO('Yoloweights\yolov8l.pt')
classnames = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
              'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
              'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
              'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
              'cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
              'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

mask = cv2.imread('mask1.png')


#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [24,200,824,200]
totalcount=[]

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100))/100
            cls = box.cls[0]
            if classnames[int(cls)] == 'car' or classnames[int(cls)] == 'bus' or classnames[int(cls)] == 'truck' or classnames[int(cls)] == 'motorbike' and conf>0.3:
                # write_text(img, x1, y1, x2, y2, classnames[int(cls)], conf)
                # cv2.putText(img, f'{classnames[int(cls)]}{conf}', (max(0,x1), max(35,y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                # cvzone.putTextRect(img, f'{classnames[int(cls)]}{conf}', (max(0,x1), max(35,y1)), scale=1, thickness=2)
                currentarray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentarray))
        resultstracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (200,95,240), 2)
        for result in resultstracker:
            x1,y1,x2,y2,id = result
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            write_text(img, x1, y1, x2, y2, id, conf, color=(255,0,0))

            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+10:
                if totalcount.count(id)==0:
                    totalcount.append(id)

        cv2.putText(img, f"Vehicle Count: {len(totalcount)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Recognition",img)
        cv2.waitKey(1)
        