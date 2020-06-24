import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', help='.xml model path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--labels', help='.txt classification labels', default='classification_classes_ILSVRC2012.txt')
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--engine', help='target engine', required=True, choices=['tf', 'opvn'])
    argv = parser.parse_args()

    if argv.engine == 'tf':
        from classification.classification_tf import TensorFlowClassification
        network = TensorFlowClassification(argv.graph)
    else:
        from classification.classification_opvn import OpenVinoClassification
        model_xml = os.path.join(argv.xml)
        model_bin = model_xml.split('.xml')[0] + '.bin'

        network = OpenVinoClassification(model_xml, model_bin)

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n')

    img = cv.imread(argv.image)  
    probs, indices = network.classify(img, top_k = 5)

    for prob, idx in zip(probs, indices):
        print(("%.4f" % prob) + ' ' + str(idx) + ' ' + labels[idx])


if __name__ == "__main__":
    main()
