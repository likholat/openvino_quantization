import tensorflow as tf
import numpy as np
import argparse
from classification import TensorFlowClassification
import cv2 as cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--databasePath', help='database path', required=True)
    argv = parser.parse_args()

    network = TensorFlowClassification(argv.graph)

    with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
        labels = f.read().strip().split('\n')  

    with open('val.txt', 'rt') as f:
        vals = f.read().strip().split('\n')

    top_1 = 0
    top_5 = 0
    size = len(vals)

    for i in range(size):
        img_path, label = vals[i].rsplit(' ')
        label = int(label)

        imgPath = argv.databasePath + img_path
        img = cv.imread(imgPath)
        val = label
        results, probability = network.classify(img, 5)

        if val == (results[0]):
            top_1 += 1

        if val in (results):
            top_5 += 1

    print()
    print('Top 1 accuracy: ' + str(top_1 / size))
    print('Top 5 accuracy: ' + str(top_5 / size))

if __name__ == "__main__":
    main()
