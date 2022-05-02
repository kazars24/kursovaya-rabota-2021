import argparse
import sys
import enum
import cv2 as cv
import time
import numpy as np
import subprocess as sp
import ffmpeg
import os

from pathlib import Path
from threading import Thread

frames_list = []
WIDTH = 0
HEIGHT = 0


class ReturnCode(enum.Enum):
    SUCCESS = 0
    CRITICAL = 1


def ffmpeg_for_threading(path_to_video, path_to_outputs):
    global frames_list, WIDTH, HEIGHT

    print('ffmpeg: video processing started')
    ffmpeg_start_time = time.time()

    try:
        probe = ffmpeg.probe(path_to_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        WIDTH = int(video_stream['width'])
        HEIGHT = int(video_stream['height'])

        command = ['ffmpeg',
                   '-i', path_to_video,
                   f'{path_to_outputs}/%03d.jpg'
                   ]
        process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)

        # Read decoded video frames from the PIPE until no more frames to read
        '''
        while True:
            # Read decoded video frame (in raw video format) from stdout process.
            buffer = process.stdout.read(WIDTH * HEIGHT * 3)

            # Break the loop if buffer length is not W*H*3 (when FFmpeg streaming ends).
            if len(buffer) != WIDTH * HEIGHT * 3:
                break

            img = np.frombuffer(buffer, np.uint8).reshape(HEIGHT, WIDTH, 3)
            frames_list.append(img)
        '''

        print("ffmpeg: processing was completed in %s seconds" % (time.time() - ffmpeg_start_time))

    except Exception as ex:
        raise ex


def ml_module(path_to_img, path_to_coordinates):
    global WIDTH, HEIGHT

    classes_for_task = ['person', 'car', 'bus', 'truck']

    net = cv.dnn_DetectionModel('ml_cfg/yolov4.cfg', 'ml_cfg/yolov4.weights')
    net.setInputSize(WIDTH - (WIDTH % 32), HEIGHT - (HEIGHT % 32))
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    frame = cv.imread(path_to_img)

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

    files = os.listdir(output_path)
    files.remove('coordinates')
    index = 1
    for image in files:
        output = f"{output_path}/coordinates/frame-{index}.txt"
        frame_start_time = time.time()
        ml_module(f"{output_path}/{image}", output)
        print(f"model: frame-{index} processing ended (%s sec)" % (time.time() - frame_start_time))
        index += 1

    print("model: processing was completed in %s seconds" % (time.time() - model_start_time))


def postproc(path_to_coordinates, roi_qp):
    print('postproc: started')
    postproc_start_time = time.time()

    folder = Path(f"{path_to_coordinates}/coordinates")
    folder_len = sum(1 for _ in folder.iterdir())

    all_regions = []
    for i in range(1, folder_len + 1):
        path_to_file = path_to_coordinates + f"/coordinates/frame-{i}.txt"
        with open(path_to_file) as f:
            objects = f.read().splitlines()
        objects = [list(map(int, x.split(' '))) for x in objects]

        regions = [f"{x[0]},{x[1]},{x[0] + x[2]},{x[1] + x[3]}:{roi_qp}" for x in objects]
        regions = ' '.join(regions)
        all_regions.append(regions)

    print("postproc: processing was completed in %s seconds" % (time.time() - postproc_start_time))

    return ' '.join(all_regions)


def encoder(path_to_video, regions, frame_num, base_qp):
    print('encoder: started')
    encoder_start_time = time.time()

    path_to_h264 = f"{path_to_video[:-4]}.h264"
    command1 = f'ffmpeg -i {path_to_video} -codec:v libx264 -pix_fmt yuv420p -preset slow -qp 10 -an {path_to_h264}'
    process1 = sp.run(command1, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

    path_to_roi = f'{path_to_h264[:-5]}_roi.h264'
    command2 = f"h264-roi-build/h264_roi {path_to_h264} {path_to_roi} -c {frame_num} -q {base_qp} {regions}"
    # print(command2)
    process2 = sp.run(command2, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

    print("encoder: processing was completed in %s seconds" % (time.time() - encoder_start_time))


def main():
    print("Starting pipeline")
    start_time = time.time()
    exit_code = ReturnCode.SUCCESS
    script = Path(__file__)

    parser = argparse.ArgumentParser(prog=script.name)
    parser.add_argument('-i', '--input', required=True, help='Path to video')
    parser.add_argument('-o', '--output', required=True, help='Output dir')
    parser.add_argument('-c', '--frame_num', required=True, help='Number of frames')
    parser.add_argument('-q', '--base_qp', required=True, help='Quantizer value from 0 to 51')
    parser.add_argument('-r', '--roi_qp', required=True, help='ROI quantizer value')

    args = parser.parse_args()
    output_path = Path(args.output)

    print(f"Input video: {args.input}")
    print(f"Output video: {f'{args.input[:-4]}_roi.h264'}")
    print(f"Base QP = {args.base_qp}")
    print(f"ROI QP = {args.roi_qp}")

    thread1 = Thread(target=ffmpeg_for_threading, args=(args.input, output_path))
    thread2 = Thread(target=ml_for_treating, args=(output_path,))

    thread1.start()
    time.sleep(0)
    thread2.start()
    thread1.join()
    thread2.join()

    regions = postproc(args.output, args.roi_qp)

    encoder(args.input, regions, args.frame_num, args.base_qp)

    print("Pipeline finished in %s seconds" % (time.time() - start_time))

    print('Original video:')
    og_stat_size = Path(args.input).stat().st_size
    print(f'- file size: {og_stat_size} bytes')
    probe = ffmpeg.probe(args.input)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    bit_rate = int(int(video_stream['bit_rate']))
    print(f'- bit rate: {bit_rate} bit/s')
    w = int(video_stream['width'])
    h = int(video_stream['height'])
    print(f'- resolution: {w}x{h}')

    print('ROI video:')
    roi_stat_size = Path(f'{args.input[:-4]}_roi.h264').stat().st_size
    print(f'- file size: {roi_stat_size} bytes')
    probe = ffmpeg.probe(f'{args.input[:-4]}_roi.h264')
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    # bit_rate = int(int(video_stream['bit_rate']))
    # print(f'- bit rate: {bit_rate} bit/s')
    w = int(video_stream['width'])
    h = int(video_stream['height'])
    print(f'- resolution: {w}x{h}')
    cmd = f"ffmpeg -i {args.input} -i {f'{args.input[:-4]}_roi.h264'} -filter_complex 'psnr' -f null /dev/null"
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    psnr = process.stdout.readlines()
    psnr = psnr[-1].decode('utf-8').split()[7]
    print('PSNR', psnr)

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
