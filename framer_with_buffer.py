import os
import argparse
import sys
import enum
import cv2 as cv
import time
import numpy as np
import subprocess as sp
import ffmpeg

from pathlib import Path
from threading import Thread


class ReturnCode(enum.Enum):
    SUCCESS = 0
    CRITICAL = 1


def ffmpeg_for_threading(path_to_video):
    try:
        # Directories and files
        filename, _ = os.path.splitext(os.path.basename(path_to_video))

        # Video params
        probe = ffmpeg.probe(path_to_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        total_frames = int(video_stream.get('nb_frames', 1000))
        duration = int(float(video_stream['duration']))
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # ffmpeg command
        command = ['ffmpeg',
                   '-i', path_to_video,
                   '-pix_fmt', 'bgr24',  # brg24 for matching OpenCV
                   '-f', 'rawvideo',
                   'pipe:']

        # Execute FFmpeg as sub-process with stdout as a pipe
        process = sp.Popen(command, stdout=sp.PIPE)

        # Read decoded video frames from the PIPE until no more frames to read
        while True:
            # Read decoded video frame (in raw video format) from stdout process.
            buffer = process.stdout.read(width * height * 3)

            # Break the loop if buffer length is not W*H*3 (when FFmpeg streaming ends).
            if len(buffer) != width * height * 3:
                break

            img = np.frombuffer(buffer, np.uint8).reshape(width, height, 3)

    except Exception as ex:
        raise ex


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


def ml_for_treating(output_path):
    if not os.path.exists(f"{output_path}/coordinates"):
        os.makedirs(f"{output_path}/coordinates")
    files = os.listdir(output_path)
    files.remove('coordinates')
    while files:
        for image in files:
            output = f"{output_path}/coordinates/{image[0:-4]}.txt"
            ml_module(f"{output_path}/{image}", output)
            print(image, 'detected!')
            os.remove(f"{output_path}/{image}")
        files = os.listdir(output_path)
        files.remove('coordinates')


def main():
    exit_code = ReturnCode.SUCCESS
    script = Path(__file__)

    parser = argparse.ArgumentParser(prog=script.name)
    parser.add_argument('-i', '--input', required=True, help='Path to video')
    parser.add_argument('-o', '--output', required=True, help='Output dir (.txt if --numpy)')

    args = parser.parse_args()
    output_path = Path(args.output)

    thread1 = Thread(target=ffmpeg_for_threading, args=(args.input,))
    thread2 = Thread(target=ml_for_treating, args=(output_path,))

    thread1.start()
    time.sleep(5)
    thread2.start()
    thread1.join()
    thread2.join()

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
