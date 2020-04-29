import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
from classification import Classification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
    parser.add_argument('--image', help='image path', required=True)
    parser.add_argument('--labels', help='.txt labels path', default='classification_classes_ILSVRC2012.txt')
    argv = parser.parse_args()

    graph = Classification.load_pb(argv.graph)

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n')  

    img = cv.imread(argv.image)

    image_mean = 127.5
    image_std = 127.5
    image_size_x = 299
    image_size_y = 299 

    img = cv.resize(img, (image_size_x, image_size_y))
    img = np.expand_dims(img, axis=0)
    img = (img - image_mean) / image_std
    img = img.astype(np.float32)

    output_layer = 'output:0'
    input_node = 'input:0'

    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph, name = "")
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: img})

    predictions = Classification.sigmoid(predictions)

    for i in np.argsort(predictions)[::-1][:5]:
        print(("%.4f" % predictions[i]) + ' ' + labels[i - 1])

if __name__ == "__main__":
    main()
