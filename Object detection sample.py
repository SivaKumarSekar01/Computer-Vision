#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install opencv-python


# In[5]:


import cv2


# 
# # object detection

# In[6]:


import cv2
from random import randint


dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
      
      
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) # freezes frame for 1ms

    match(key):
        case 27: # esc key to exit
            capture.release()
            cv2.destroyAllWindows()

        case 13: # enter key to reset colors
            color_map = {}
    
        

        


# In[ ]:


import cv2
from random import randint

dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate the width and height of the bounding box
        box_width = w
        box_height = h

        # Draw rectangle and text
        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Measure length and breadth
        cv2.putText(frame, f'Length: {box_width} pixels', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.putText(frame, f'Breadth: {box_height} pixels', (x, y + h + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) # freezes frame for 1ms

    if key == 27: # esc key to exit
        break

    if key == 13: # enter key to reset colors
        color_map = {}

capture.release()
cv2.destroyAllWindows()


# # object detection with length and breadth

# In[1]:


import cv2
from random import randint

# Define your conversion factor (pixels to centimeters)
conversion_factor = 0.1  # This is just an example, you need to replace it with your actual conversion factor

dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate the width and height of the bounding box
        box_width_cm = w * conversion_factor
        box_height_cm = h * conversion_factor

        # Draw rectangle and text
        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Measure length and breadth
        cv2.putText(frame, f'Length: {box_width_cm} cm', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.putText(frame, f'Breadth: {box_height_cm} cm', (x, y + h + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) # freezes frame for 1ms

    if key == 27: # esc key to exit
        break

    if key == 13: # enter key to reset colors
        color_map = {}

capture.release()
cv2.destroyAllWindows()


# 

# In[ ]:


import cv2
import numpy as np

# Camera intrinsic parameters (focal length and principal point)
focal_length = 800  # Example value, replace with actual focal length in pixels
principal_point = (640, 360)  # Example value, replace with actual principal point

# Known dimensions of reference object
known_width = 20  # Width of reference object in centimeters
known_distance = 100  # Distance of reference object from camera in centimeters

# Function to calculate distance from camera to object
def calculate_distance(known_width, focal_length, measured_width):
    return (known_width * focal_length) / measured_width

# Function to measure object dimensions
def measure_object(image, box):
    x, y, w, h = box
    measured_width = w
    measured_height = h
    distance = calculate_distance(known_width, focal_length, measured_width)
    return measured_width, measured_height, distance

# Load YOLO model and classes
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
classes = None
with open('classes.txt', 'r') as f:
    classes = f.read().strip().split('\n')

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward()

    # Process detections
    for detection in outs:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                measured_width, measured_height, distance = measure_object(frame, (x, y, w, h))

                # Draw bounding box and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{classes[class_id]} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'Width: {measured_width:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'Height: {measured_height:.2f} cm', (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# # AREA

# In[2]:


import cv2
from random import randint

# Define your conversion factor (pixels to centimeters)
conversion_factor = 0.1  # This is just an example, you need to replace it with your actual conversion factor

dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate the area of the bounding box
        box_area_cm2 = w * conversion_factor * h * conversion_factor

        # Draw rectangle and text
        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display area
        cv2.putText(frame, f'Area: {box_area_cm2} cm^2', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) # freezes frame for 1ms

    if key == 27: # esc key to exit
        break

    if key == 13: # enter key to reset colors
        color_map = {}

capture.release()
cv2.destroyAllWindows()


# # DIAMETER

# In[ ]:


import cv2
from random import randint

# Define your conversion factor (pixels to centimeters)
conversion_factor = 0.1  # This is just an example, you need to replace it with your actual conversion factor

dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read() 
    frame = cv2.flip(frame,1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate the diameter of the object (using width as diameter)
        diameter_cm = w * conversion_factor

        # Draw rectangle and text
        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display diameter
        cv2.putText(frame, f'Diameter: {diameter_cm} cm', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1) # freezes frame for 1ms

    if key == 27: # esc key to exit
        break

    if key == 13: # enter key to reset colors
        color_map = {}

capture.release()
cv2.destroyAllWindows()


# # testing

# In[7]:


import cv2
from random import randint

# Define your conversion factor (pixels to centimeters) - Initially set to 1
conversion_factor = 1.0  # Replace with your calculated value

# Function to calculate and display object dimensions
def calculate_dimensions(frame, box, color):
    x, y, w, h = box
    box_width_cm = w * conversion_factor
    box_height_cm = h * conversion_factor
    cv2.putText(frame, f'Length: {box_width_cm:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    cv2.putText(frame, f'Breadth: {box_height_cm:.2f} cm', (x, y + h + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

# Load YOLOv4 model
dnn = cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load class names
with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

# Capture video
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Color map for bounding boxes
color_map = {}

while True:
    # Capture frame
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # Object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate and display object dimensions (using the function)
        calculate_dimensions(frame, box, color_map.get(obj_class, (randint(0, 255), randint(0, 255), randint(0, 255))))
        color_map[obj_class] = color  # Update color map

        # Draw bounding box and class label
        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display frame
    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1)  # wait for 1 millisecond

    # Exit on ESC key press
    if key == 27:
        break

    # Reset color map on ENTER key press
    if key == 13:
        color_map = {}

# Release resources
capture.release()
cv2.destroyAllWindows()


# In[8]:


pip install opencv-contrib-python


# # Angle

# In[3]:


import cv2
import math
from random import randint

# Define your conversion factor (pixels to centimeters)
conversion_factor = 0.1  # This is just an example, you need to replace it with your actual conversion factor

dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

color_map = {}

while True:
    # frame capture
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # object detection
    class_ids, confidences, boxes = model.detect(frame)
    for id, confidence, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        obj_class = classes[id]

        # Calculate the width and height of the bounding box
        box_width_cm = w * conversion_factor
        box_height_cm = h * conversion_factor

        # Calculate the angle
        angle = math.degrees(math.atan(h / w))

        # Draw rectangle and text
        if obj_class not in color_map:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            color_map[obj_class] = color
        else:
            color = color_map[obj_class]

        cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Measure angle
        cv2.putText(frame, f'Angle: {angle} degrees', (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    cv2.imshow('Video Capture', frame)
    key = cv2.waitKey(1)  # freezes frame for 1ms

    if key == 27:  # esc key to exit
        break

    if key == 13:  # enter key to reset colors
        color_map = {}

capture.release()
cv2.destroyAllWindows()


# In[ ]:




