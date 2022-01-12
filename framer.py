import os
import argparse
import sys
import enum
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

        for sec in range(duration):
            out, err = (
                ffmpeg
                    .input(args.input, ss=sec)
                    # .filter_('select', 'gte(n,{})'.format(frame_num))
                    .output(f"{args.output}/{filename}-{sec}.png", vframes=1, format='image2', vcodec='mjpeg')
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

    files = os.listdir(output_path)
    for image in files:
        output = f"{output_path}/{image[0:-4]}.txt"
        ml_module(f"{output_path}/{image}", output)
        print(image, 'detected!')

    return exit_code.value


if __name__ == '__main__':
    sys.exit(main())
