import cv2
import numpy as np
import time
import os
import argparse

# path
modelConfiguration = 'model/yolov3_training.cfg'
modelWeights = 'model/yolov3_training_4000.weights'
classesFile = 'model/Plate.names'

classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# Function Find and Detect Object
def findObjects(outputs,img,img_id):

    confThreshold = 0.5
    nmsThreshold = 0.3
    imageid = img_id

    hT ,wT ,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in indices:
        check_laber = False
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        label = str(classNames[classIds[i]])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Detect and cut image if not have helmet
        try:
            if (label == "Person"):
                print("Crop Person :", imageid)
                crop = img[y:y+h, x:x+w]
                cv2.imwrite("person_crop/person_"+str(imageid)+".jpg",crop)
                check_laber = True
                imageid += 1
        except:
            print("False imwrite")
    
    return imageid

def person_detect(pathUrl):

    whT = 320

    print("------------- Program Start -------------")
    print("---------- Detect Motorcyclist ----------")

    # cap = cv2.VideoCapture(pathUrl["VideoPath"])
    cap = cv2.VideoCapture(pathUrl)
    print("URL: ",pathUrl)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    print("This video have :", fps, " Frame")

    # className Here<Edit>
    # setting
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #
    speed = 0
    img_id = 0
    complete = False
    # start_time = time.time()
    while (complete == False):
        try:
            _, img = cap.read()
            # Video frame skip
            if (speed % 40 == 0):
                blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layerNames = net.getLayerNames()
                outputNames = [layerNames[i[0]-1]for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                imageid = findObjects(outputs, img, img_id)
                img_id = imageid
                speed += 1
                key = cv2.waitKey(1)
            else:
                speed += 1
                key = cv2.waitKey(1)
        except:
            print("_____________ End Process 1 _____________")
            complete = True
            cap.release()
            cv2.destroyAllWindows()
