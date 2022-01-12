import cv2 as cv


def ml_module(path_to_img, path_to_coordinates):
    classes_for_task = ['person', 'car', 'bus', 'truck']

    net = cv.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    frame = cv.imread(path_to_img)

    with open('coco.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)

    objects = [[names[x[0]], x[1]] for x in zip(classes.flatten(), boxes)]
    for x in objects:
        if x[0] not in classes_for_task:
            objects.remove(x)

    with open(path_to_coordinates, "w") as f:
        f.write("\n".join(" ".join(map(str, x[1])) for x in objects))
