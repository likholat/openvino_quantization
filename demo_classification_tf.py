import tensorflow as tf
import numpy as np
import cv2
import argparse

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help=".pb graph path")
    parser.add_argument("image", help="image path")
    parser.add_argument("labels", help=".txt labels path")
    argv = parser.parse_args()

    graph = load_pb(argv.graph)

    labels = []
    labels_filename = argv.labels

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    imageFile = argv.image
    image = cv2.imread(imageFile)

    image_mean = 127.5
    image_std = 127.5
    image_size_x = 299
    image_size_y = 299 

    image = cv2.resize(image, (image_size_x, image_size_y), interpolation = cv2.INTER_LINEAR)
    img = image.reshape((1, 299, 299, 3))
    img = (img - image_mean) / image_std

    img = np.array(img, dtype=float)

    output_layer = 'output:0'
    input_node = 'input:0'

    with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph.as_graph_def(), name = "")
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: img})
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)

    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    print()
    print(highest_probability_index)
    print('Classified as: ' + labels[highest_probability_index])
    print()

if __name__ == "__main__":
    main()
