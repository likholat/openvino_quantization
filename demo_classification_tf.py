import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
from classification.classification_tf import TensorFlowClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--labels', help='.txt classification labels', default='classification_classes_ILSVRC2012.txt')
    parser.add_argument('--image', help='image path', required=True)
    argv = parser.parse_args()

    network = TensorFlowClassification(argv.graph)

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n') 

    img = cv.imread(argv.image) 

    probs, indices = network.classify(img, top_k = 5)

    for prob, idx in zip(probs, indices):
        print(("%.4f" % prob) + ' ' + str(idx) + ' ' + labels[idx])

if __name__ == "__main__":
    main()
