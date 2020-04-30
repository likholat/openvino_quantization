import tensorflow as tf
import numpy as np
import cv2 as cv

class Classification:
    graph = None

    def __init__(self, path_to_pb): 
        with tf.io.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.graph = graph_def

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def classify(self, img):
        image_mean = 127.5
        image_std = 127.5
        image_size_x = 299
        image_size_y = 299 

        output_layer = 'output:0'
        input_node = 'input:0'

        with tf.compat.v1.Session() as sess:
            tf.import_graph_def(self.graph, name = "")
            g = self.graph
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)

            img = cv.resize(img, (image_size_x, image_size_y))
            img = np.expand_dims(img, axis=0)
            img = (img - image_mean) / image_std
            img = img.astype(np.float32)

            predictions, = sess.run(prob_tensor, {input_node: img})
            predictions = Classification.sigmoid(predictions)

            return np.argsort(predictions)[::-1][:5]