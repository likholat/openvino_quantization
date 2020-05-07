import tensorflow as tf
import numpy as np
import cv2 as cv

class TensorFlowClassification:
    def __init__(self, path_to_pb): 
        with tf.io.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.graph = graph_def
            tf.import_graph_def(self.graph, name = "")
            self.sess = tf.compat.v1.Session()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify(self, img, top_k):
        image_mean = 127.5
        image_std = 127.5
        image_size_x = 299
        image_size_y = 299 

        output_layer = 'output:0'
        input_node = 'input:0'
        
        prob_tensor = self.sess.graph.get_tensor_by_name(output_layer)

        img = cv.resize(img, (image_size_x, image_size_y))
        img = np.expand_dims(img, axis=0)
        img = (img - image_mean) / image_std
        img = img.astype(np.float32)

        predictions, = self.sess.run(prob_tensor, {input_node: img})
        predictions = self.sigmoid(predictions)

        return np.sort(predictions)[::-1][:top_k], np.argsort(predictions)[::-1][:top_k]
