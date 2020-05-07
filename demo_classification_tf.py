import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
from classification import TensorFlowClassification

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

    top_k = 5
    probability, result= network.classify(img, top_k)

    for i in range(top_k):
        print(("%.4f" % probability[i]) + ' ' + str(result[i]) + ' ' + labels[result[i] - 1])
    # we need -1 for: labels[result[i] - 1], because classification_classes_ILSVRC2012.txt starts with 1, our labels starts with 0

if __name__ == "__main__":
    main()
