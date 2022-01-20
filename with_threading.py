from threading import Thread, Lock
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import sys
if sys.version > '3':
	import queue as Queue
else:
	import Queue

##global image,w,h
##image = []

##cap = cv2.VideoCapture(0)



class WebcamVideoStream : 
    global frame
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
##        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
##        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()    
            
def main():
    global result,flag,image,frame,text,x,y,Image,overlay,box_coords,idxs,ww,yy,hh,xx
    while True:
##        image=img
        print("Working2")
        flag=0
##        _, image = cap.read()
        
        h, w= image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.perf_counter()
        layer_outputs = net.forward(ln)
        time_took = time.perf_counter() - start
        print("Time took:", time_took)
        boxes, confidences, class_ids = [], [], []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the object detections
            for detection in output:
                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # discard weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    

        # perform the non maximum suppression given the scores defined before
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        font_scale = 1
        thickness = 1

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                xx, yy = boxes[i][0], boxes[i][1]
                ww, hh = boxes[i][2], boxes[i][3]
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(Image, (xx, yy), (xx + ww, yy + hh), color=color, thickness=-1)
                text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
                print(text)
                if(text=="Plastic" and confidance > 0.7):
                        p.ChangeDutyCycle(5)
                        time.sleep(0.5)
                        p.ChangeDutyCycle(7.5)
                        time.sleep(0.5)
                # calculate text width & height to draw the transparent boxes as background of the text
##                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
##                text_offset_x = x
##                text_offset_y = y - 5
##                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
##                overlay = image.copy()
##                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
##                Image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # now put the text (label: confidence %)
##                cv2.putText(Image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
##                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                

        

if __name__=='__main__':
     servoPIN = 17
     GPIO.setmode(GPIO.BCM)
     GPIO.setup(servoPIN, GPIO.OUT)

    p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
    p.start(2.5) # Initialization   
    global image,result,flag,frame,text,x,y,Image,overlay,box_coords,idxs,ww,yy,hh,xx
    box_coords = {0,0,0}
##    Image=np.zeros([500,500,3],dtype=np.uint8)
##    idxs = np.empty((5, 5))
    flag = 0
    x =0
    y =0
    xx,yy,hh,ww=0,0,0,0
    text =" "
##    frame =np.zeros([500,500,3],dtype=np.uint8)
##    image = np.zeros([500,500,3],dtype=np.uint8)
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = "cfg/yolov3.cfg"
    weights_path = "weights/yolov3.weights"
    font_scale = 1
    thickness = 1
    LABELS = ["Plastic"]
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    vs = WebcamVideoStream().start()
    t1 = Thread(target = main , args=())
    t1.start()
####    t1.join()
    while True :
        
        image = vs.read()
##        if(len(idxs)>0:
##           for i in idxs.flatten():
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(image, box_coords[0], box_coords[1], color=(0, 0, 0), thickness=cv2.FILLED)
        cv2.rectangle(image, (xx, yy), (xx + ww, yy + hh), color=(0,0,1), thickness=1)

        Image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        cv2.putText(Image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        cv2.imshow('webcam', Image)

        if cv2.waitKey(1) == 27 :
            break

    vs.stop()
    cv2.destroyAllWindows()
    p.stop()
    GPIO.cleanup()



