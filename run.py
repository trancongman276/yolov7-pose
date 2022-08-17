import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

from utils.datasets import letterbox
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def parse_arg(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to image', default=None, required=True)
    parser.add_argument('--output_path', type=str, help='Path to output', default=None, required=True)
    parser.add_argument('--model_path', type=str, help='Path to model',
                        default='./weights/yolov7-w6-pose-nms.onnx')
    return parser.parse_args(argv)


def preprocess(img):
    # Resize + Padding to size (960, 960, 3)
    img, r, d = letterbox(img, 960, stride=64, auto=False)
    # Reshape to size (1, 3, 960, 960)
    _img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    # Normalize image
    _img = _img / 255.
    _img = _img.astype(np.float32)
    return {'resized': img, 'norm':_img}


def postprocess(img, ort_outs):
    # Get skeleton key-points
    output_kpt = output_to_keypoint(ort_outs)
    img_out = img.copy()
    # Draw skeleton to image
    for idx in range(output_kpt.shape[0]):
        plot_skeleton_kpts(img_out, output_kpt[idx, 7:].T, 3)
    return img_out


def predict(img):
    # PreProcess image
    meta = preprocess(img)
    # Predict
    ort_inputs = {model.get_inputs()[0].name: meta['norm']}
    ort_outs = model.run(['output'], ort_inputs)
    # PostProcess result
    img_out = postprocess(meta['resized'], ort_outs)
    return img_out


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    # Init model
    print('Init model')
    model = ort.InferenceSession(args.model_path)
    # Input image
    print('Input image')
    img = cv2.imread(args.image_path)
    # Predict
    print('Predicting')
    img_out = predict(img)
    # Write down result
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, 'result.jpg')
    print('Write down results to: ', output_path)
    cv2.imwrite(output_path, img_out)
