import os
import argparse
import sys
import enum
import numpy as np
import cv2 as cv

import ffmpeg

from pathlib import Path


class ReturnCode(enum.Enum):
    SUCCESS = 0
    CRITICAL = 1


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


def main():
    exit_code = ReturnCode.SUCCESS
    script = Path(__file__)

    parser = argparse.ArgumentParser(prog=script.name)
    parser.add_argument('-i', '--input', required=True, help='Path to video')
    parser.add_argument('-o', '--output', required=True, help='Output dir (.txt if --numpy)')
    parser.add_argument('-np', '--numpy', required=False, action='store_true', help='Transform to nparray')

    args = parser.parse_args()
    try:
        # Directories and files
        output_path = Path(args.output)
        filename, _ = os.path.splitext(os.path.basename(args.input))

        # Video params
        probe = ffmpeg.probe(args.input)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        total_frames = int(video_stream.get('nb_frames', 1000))
        duration = int(float(video_stream['duration']))
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        if args.numpy:
            out, _ = (
                ffmpeg
                    .input(args.input)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True)
            )
            video = (
                np
                    .frombuffer(out, np.uint8)
                    .reshape([-1, height, width, 3])
            )
            if output_path.is_dir():
                output_path = output_path / f"{filename}.txt"
            with open(output_path, 'w+') as handle:
                handle.write(video)
        else:
            for sec in range(duration):
                out, err = (
                    ffmpeg
                        .input(args.input, ss=sec)
                        # .filter_('select', 'gte(n,{})'.format(frame_num))
                        .output(f"{args.output}/{filename}-{sec}.jpg", vframes=1, format='image2', vcodec='mjpeg')
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
        raise (ex)
        exit_code = ReturnCode.CRITICAL

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
