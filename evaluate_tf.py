import tensorflow as tf
import numpy as np
import argparse
from classification import TensorFlowClassification
import cv2 as cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--databasePath', help='database path', required=True)
    parser.add_argument('--rightVals', help='.txt val path', default='val.txt')
    argv = parser.parse_args()

    network = TensorFlowClassification(argv.graph)

    with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
        labels = f.read().strip().split('\n')  

    
    with open(argv.rightVals, 'rt') as f:
        vals = f.read().strip().split('\n')

    for val in vals:
        img_path, label = val.rsplit(' ')
        label = int(label)

    top_1 = 0
    top_5 = 0
    # size = len(vals)
    size = 10

    for i in range(size):
        imgPath = argv.databasePath + vals[i].rsplit(' ')[0]
        img = cv.imread(imgPath)
        val = int(vals[i].rsplit(' ')[1])
        results = network.classify(img, 5)

        if val == (results[0][0] - 1):
            top_1 += 1

        if val in (results[0] - 1):
            top_5 += 1

    print()
    print('Top 1 accuracy: ' + str(top_1 / size))
    print('Top 5 accuracy: ' + str(top_5 / size))

if __name__ == "__main__":
    main()
