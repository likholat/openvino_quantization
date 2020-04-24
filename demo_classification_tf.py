import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='.pb graph path', default = 'resnet_v2_101_299_frozen.pb')
    parser.add_argument('--image', help='image path', default = 'realistic-blue-umbrella_1284-11412.jpg')
    parser.add_argument('--labels', help='.txt labels path', default = 'classification_classes_ILSVRC2012.txt')
    argv = parser.parse_args()

    graph = load_pb(argv.graph)

    with open(argv.labels, 'rt') as f:
        labels = f.read().strip().split('\n')  

    img = cv.imread(argv.image)

    image_mean = 127.5
    image_std = 127.5
    image_size_x = 299
    image_size_y = 299 

    img = cv.resize(img, (image_size_x, image_size_y), interpolation = cv.INTER_LINEAR)
    img = np.expand_dims(img, axis = 0)
    img = (img - image_mean) / image_std
    img = img.astype(np.float32)

    output_layer = 'output:0'
    input_node = 'input:0'

    with tf.compat.v1.Session() as sess:
        # sess.graph.as_default()
        tf.import_graph_def(graph.as_graph_def(), name = "")
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: img})

    predictions = sigmoid(predictions)

    for i in np.argsort(-1*predictions)[:5]:
        print(("%.4f" % predictions[i]) + ' ' + labels[i - 1])

if __name__ == "__main__":
    main()
