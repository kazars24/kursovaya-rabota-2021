import os
import argparse
import sys
import enum
import cv2 as cv
import time

import ffmpeg

from pathlib import Path
from threading import Thread


class ReturnCode(enum.Enum):
    SUCCESS = 0
    CRITICAL = 1


def ffmpeg_for_threading(path_to_video, path_to_outputs):
    try:
        # Directories and files
        output_path = Path(path_to_outputs)
        filename, _ = os.path.splitext(os.path.basename(path_to_video))

        # Video params
        probe = ffmpeg.probe(path_to_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        total_frames = int(video_stream.get('nb_frames', 1000))
        duration = int(float(video_stream['duration']))
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        for sec in range(duration):
            out, err = (
                ffmpeg
                    .input(path_to_video, ss=sec)
                    # .filter_('select', 'gte(n,{})'.format(frame_num))
                    .output(f"{path_to_outputs}/{filename}-{sec}.png", vframes=1, format='image2', vcodec='mjpeg')
                    .run(capture_stdout=True)
            )
            print(err)
            """out, _ = (
                ffmpeg
                .input(args.input)
                .filter('fps')
                .output(f"{args.output}/{filename}-%d.jpg", start_number=0, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True)
             )"""
    except Exception as ex:
        raise ex
        exit_code = ReturnCode.CRITICAL


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

    thread1 = Thread(target=ffmpeg_for_threading, args=(args.input, args.output))
    thread2 = Thread(target=ml_for_treating, args=(output_path,))

    thread1.start()
    time.sleep(5)
    thread2.start()
    thread1.join()
    thread2.join()

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
