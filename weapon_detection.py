import cv2
import numpy as np
import os

base_dir = r"E:\BSBA\6th Sem\Artificial Intelligence\Project\Weapon-Detection-with-yolov3-master\weapon_detection"


weights_path = os.path.join(base_dir, "yolov3_training_2000.weights")
cfg_path = os.path.join(base_dir, "yolov3_testing.cfg")
net = cv2.dnn.readNet(weights_path, cfg_path)


if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
    print("Error: Weights or cfg file not found.")
    exit()

classes = ["Weapon"]


output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def value():
    val = input("Enter file name or press enter to start CCTV_cam: \n")
    if val == "":
        val = 0
    return val


cap = cv2.VideoCapture(value())

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to read a frame from the video source.")
        break
    height, width, channels = img.shape

    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0: 
        print("Weapon detected in frame")
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # Stop if 'Esc' or 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
