import cv2
import numpy as np
import time
import glob
import os
import random
from PersonDetect import findObjects


def helmet_detect():
    print("------------- Detect Helmet -------------")

    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3

    modelConfiguration = 'model/yolov3_training.cfg'
    modelWeights = 'model/yolov3_training_4000.weights'
    classesFile = 'model/Plate.names'
    
    # classfile
    classNames = []
    with open(classesFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load Yolo
    net = cv2.dnn.readNet(modelWeights,modelConfiguration)

    # Name custom object
    classes = ["Helmet","Person","Plate"]

    # Images path
    images_path = glob.glob("person_crop/*.jpg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        head_tail = os.path.split(img_path)

        x = head_tail[1].split("_")
        y = x[1].split(".")
        z = y[0]
        # Loading image
        img = cv2.imread(img_path)
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        check_helmet = False
        check_plate = False

        for i in range(len(boxes)):

            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                
                if (label == "Helmet"):
                    check_helmet = True
                    
                if (label == "Plate" and check_helmet == False):
                    cv2.imwrite("person_nohelmet/person_"+str(z)+".jpg",img)
                
        key = cv2.waitKey(1)
    print("_____________ End Process 2 ______________")
    cv2.destroyAllWindows()





