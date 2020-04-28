import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
import os
from os import listdir
from os.path import isfile, join


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def

def find_right_val(image_name, list):
        for text in list: 
            if image_name in text: 
                parts = text.rsplit(' ')
                return int(parts[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--labels', help='.txt labels path', default='classification_classes_ILSVRC2012.txt')
    parser.add_argument('--databasePath', help='database path', required=True)
    parser.add_argument('--rightVals', help='.txt val path', default='val.txt')
    argv = parser.parse_args()

    graph = load_pb(argv.graph)
    images = [f for f in listdir(argv.databasePath) if isfile(join(argv.databasePath, f))]

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n')  

    with open(argv.rightVals, 'rt') as f:
        vals = f.read().strip().split('\n')

    image_mean = 127.5
    image_std = 127.5
    image_size_x = 299
    image_size_y = 299 

    output_layer = 'output:0'
    input_node = 'input:0'

    top_1 = 0
    top_5 = 0
    # size = len(images)
    size = 10

    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph, name = "")
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)

        for i in range(size):
            if (i % 1000) == 0:
                print(str(i))

            imgPath = argv.databasePath + images[i]
            img = cv.imread(imgPath)

            img = cv.resize(img, (image_size_x, image_size_y))
            img = np.expand_dims(img, axis=0)
            img = (img - image_mean) / image_std
            img = img.astype(np.float32)

            predictions, = sess.run(prob_tensor, {input_node: img})
            predictions = sigmoid(predictions)

            val = find_right_val(images[i], vals)
            results = np.argsort(predictions)[::-1][:5]

            if val == (results[0] - 1): top_1 += 1

            if val in (results - 1): top_5 += 1

    top_1 = top_1 / size
    top_5 = top_5 / size

    print()
    print('Top 1 accuracy: ' + str(top_1))
    print('Top 5 accuracy: ' + str(top_5))

if __name__ == "__main__":
    main()
