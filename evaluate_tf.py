import tensorflow as tf
import numpy as np
import os
import argparse
from classification import TensorFlowClassification
import cv2 as cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--dataset', help='database path', required=True)
    argv = parser.parse_args()

    network = TensorFlowClassification(argv.graph)

    with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
        labels = f.read().strip().split('\n')  

    with open(os.path.join(argv.dataset, 'val.txt'), 'rt') as f:
        vals = f.read().strip().split('\n')

    top_1 = 0
    top_5 = 0

    for value in vals:
        img_path, label = value.rsplit(' ')
        label = int(label) + 1

        imgPath = os.path.join(argv.dataset, img_path)
        img = cv.imread(imgPath)
        probability, results = network.classify(img, 5)

        if label == (results[0]):
            top_1 += 1

        if label in (results):
            top_5 += 1

    print()
    print('Top 1 accuracy: ' + str(top_1 / len(vals)))
    print('Top 5 accuracy: ' + str(top_5 / len(vals)))

if __name__ == "__main__":
    main()
