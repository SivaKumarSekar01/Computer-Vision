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
