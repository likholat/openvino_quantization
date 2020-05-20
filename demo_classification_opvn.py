import numpy as np
import cv2 as cv
import argparse
import os

from classification.classification_opvn import OpenVinoClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.xml graph path', default='resnet_v2_101_299_opt.xml')
    parser.add_argument('--labels', help='.txt classification labels', default='classification_classes_ILSVRC2012.txt')
    parser.add_argument('--image', help='image path', required=True)
    argv = parser.parse_args()

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n') 

    img = cv.imread(argv.image) 

    model_xml = os.path.join(argv.graph, 'resnet_v2_101_299_opt.xml')
    model_bin = os.path.join(argv.graph, 'resnet_v2_101_299_opt.bin')

    network = OpenVinoClassification(model_xml, model_bin)

    probs, indices = network.classify(img, top_k = 5)

    for prob, idx in zip(probs, indices):
        print(("%.4f" % prob) + ' ' + str(idx) + ' ' + labels[idx])


if __name__ == "__main__":
    main()
