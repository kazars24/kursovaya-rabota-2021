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

frames_list = []
WIDTH = 0
HEIGHT = 0


class ReturnCode(enum.Enum):
    SUCCESS = 0
    CRITICAL = 1


def ffmpeg_for_threading(path_to_video):
    global frames_list, WIDTH, HEIGHT

    print('ffmpeg: video processing started')
    ffmpeg_start_time = time.time()

    try:
        # Directories and files
        # filename, _ = os.path.splitext(os.path.basename(path_to_video))

        # Video params
        probe = ffmpeg.probe(path_to_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        WIDTH = int(video_stream['width'])
        HEIGHT = int(video_stream['height'])

        # ffmpeg command
        command = ['ffmpeg',
                   '-i', path_to_video,
                   '-pix_fmt', 'bgr24',  # brg24 for matching OpenCV
                   '-f', 'rawvideo',
                   'pipe:']

        # Execute FFmpeg as sub-process with stdout as a pipe
        process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)  # stdout

        # Read decoded video frames from the PIPE until no more frames to read
        while True:
            # Read decoded video frame (in raw video format) from stdout process.
            buffer = process.stdout.read(WIDTH * HEIGHT * 3)

            # Break the loop if buffer length is not W*H*3 (when FFmpeg streaming ends).
            if len(buffer) != WIDTH * HEIGHT * 3:
                break

            img = np.frombuffer(buffer, np.uint8).reshape(HEIGHT, WIDTH, 3)
            frames_list.append(img)

        print("ffmpeg: processing was completed in %s seconds" % (time.time() - ffmpeg_start_time))

    except Exception as ex:
        raise ex


def ml_module(frame, path_to_coordinates):
    global WIDTH, HEIGHT

    classes_for_task = ['person', 'car', 'bus', 'truck']

    net = cv.dnn_DetectionModel('ml_cfg/yolov4.cfg', 'ml_cfg/yolov4.weights')
    net.setInputSize(WIDTH - (WIDTH % 32), HEIGHT - (HEIGHT % 32))
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    with open('ml_cfg/coco.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.4, nmsThreshold=0.4)

    objects = [[names[x[0]], x[1]] for x in zip(classes.flatten(), boxes)]
    for x in objects:
        if x[0] not in classes_for_task:
            objects.remove(x)

    with open(path_to_coordinates, "w") as f:
        f.write("\n".join(" ".join(map(str, x[1])) for x in objects))


def ml_for_treating(output_path):
    global frames_list

    print('model: frame processing started')
    model_start_time = time.time()

    # path = Path(output_path)
    coordinates_path = Path(f"{output_path}/coordinates")
    if not coordinates_path.exists():
        coordinates_path.mkdir(parents=True)

    frame_iter = iter(frames_list)
    index = 0
    while True:
        frame = next(frame_iter, False)
        if isinstance(frame, np.ndarray):
            output = f"{output_path}/coordinates/frame-{index}.txt"
            frame_start_time = time.time()
            ml_module(frame, output)
            print(f"model: frame-{index} processing ended (%s sec)" % (time.time() - frame_start_time))
            index += 1
        else:
            break

    print("model: processing was completed in %s seconds" % (time.time() - model_start_time))


def postproc(path_to_coordinates, base_qp):
    print('postproc: started')
    postproc_start_time = time.time()

    folder = Path(f"{path_to_coordinates}/coordinates")
    folder_len = sum(1 for _ in folder.iterdir())
    all_regions = []
    for i in range(folder_len):
        path_to_file = path_to_coordinates + f"/coordinates/frame-{i}.txt"
        with open(path_to_file) as f:
            objects = f.read().splitlines()
        objects = [list(map(int, x.split(' '))) for x in objects]
        regions = [f"{x[0]},{x[1] - x[3]},{x[0] + x[2]},{x[1]}:{base_qp}" for x in objects]
        regions = ' '.join(regions)
        all_regions.append(regions)

    print("postproc: processing was completed in %s seconds" % (time.time() - postproc_start_time))

    return ' '.join(all_regions)


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

    regions = postproc(args.output, 51)
    print(regions)

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
