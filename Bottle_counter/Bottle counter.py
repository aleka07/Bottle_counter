import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8x.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('1234.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
tracker=Tracker()
cx1=350 # Changed from cy1 to cx1 for x-axis tracking
offset=6 
counter=[]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(1280,720))

    results=model.predict(frame)

    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")

    list=[]         
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        if d == 39:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,200),2)
        cv2.circle(frame, (cx,cy),3,(255,0,255),-1)
        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if cx1>(cx-offset) and cx1<(cx+offset): # Changed from y4 to x4 for x-axis tracking
            if counter.count(id)==0:
                counter.append(id)
    
    cv2.line(frame,(cx1,258),(cx1,720),(0,255,0),2) # Changed from cy1 to cx1 for x-axis tracking
    l = (len(counter))
    cv2.putText(frame, f'Bottles: {l}', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

# Writing the total count into a txt file
with open("count.txt", "w") as file:
    file.write(f"The total count of bottles is {len(counter)}.\n")

cap.release()
cv2.destroyAllWindows()