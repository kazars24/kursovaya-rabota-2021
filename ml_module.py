import cv2 as cv

# created a model from the YOLOv4 configuration and pre-trained weights file
net = cv.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')

#  image preprocessing for this model
net.setInputSize(704, 704)  # resizing
net.setInputScale(1.0 / 255)  # normalizing
net.setInputSwapRB(True)  # maintaining the RGB format

frame = cv.imread('for_test.jpeg')

with open('coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)

'''
for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    label = '%.2f' % confidence
    label = '%s: %s' % (names[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left, top, width, height = box
    top = max(top, labelSize[1])
    cv.rectangle(frame, box, color=(0, 255, 0), thickness=2)
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    cv.imshow('out', frame)
'''

with open('output.txt', "w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in boxes))
