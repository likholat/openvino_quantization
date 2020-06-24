import tensorflow as tf
import numpy as np
import os
import argparse
import cv2 as cv

from classification.classification_tf import TensorFlowClassification
from classification.classification_opvn import OpenVinoClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--dataset', help='database path', required=True)
    parser.add_argument('--engine', help='target engine', required=True, choices=['tf', 'opvn'])
    argv = parser.parse_args()

    with open(os.path.join(argv.dataset, 'val.txt'), 'rt') as f:
        vals = f.read().strip().split('\n')

    top_1 = 0
    top_5 = 0

    if (argv.engine == 'tf'):
        network = TensorFlowClassification(argv.graph)
        
    elif (argv.engine == 'opvn'):
        model_xml = os.path.join(argv.graph, 'resnet_v2_101_299_opt.xml')
        model_bin = os.path.join(argv.graph, 'resnet_v2_101_299_opt.bin')

        network = OpenVinoClassification(model_xml, model_bin)

    for i, value in enumerate(vals):
        if(i % 10 == 0):
            print(i)

        img_path, label = value.rsplit(' ')
        label = int(label) + 1

        imgPath = os.path.join(argv.dataset, img_path)
        img = cv.imread(imgPath)
        probability, results = network.classify(img, 5)

        if label == results[0]:
            top_1 += 1

        if label in results:
            top_5 += 1


    print()
    print('Top 1 accuracy: ' + str(top_1 / len(vals)))
    print('Top 5 accuracy: ' + str(top_5 / len(vals)))

if __name__ == "__main__":
    main()