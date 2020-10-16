import tensorflow as tf
import numpy as np
import os
import argparse
import cv2 as cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb or .xml model path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--dataset', help='database path', required=True)
    parser.add_argument('--engine', help='target engine', required=True, choices=['tf', 'opvn'])
    argv = parser.parse_args()

    if argv.engine == 'tf':
        from classification.classification_tf import TensorFlowClassification
        network = TensorFlowClassification(argv.graph)
    else:
        from classification.classification_opvn import OpenVinoClassification
        model_xml = os.path.join(argv.graph)
        model_bin = model_xml.split('.xml')[0] + '.bin'

        network = OpenVinoClassification(model_xml, model_bin)

    with open(os.path.join(argv.dataset, 'val.txt'), 'rt') as f:
        vals = f.read().strip().split('\n')

    top_1 = 0
    top_5 = 0

    for i, value in enumerate(vals):
        if(i % 1000 == 0):
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
