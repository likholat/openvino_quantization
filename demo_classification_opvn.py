import openvino
import numpy as np
import cv2 as cv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--labels', help='.txt classification labels', default='classification_classes_ILSVRC2012.txt')
    parser.add_argument('--image', help='image path', required=True)
    argv = parser.parse_args()

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n') 

    img = cv.imread(argv.image) 

if __name__ == "__main__":
    main()
