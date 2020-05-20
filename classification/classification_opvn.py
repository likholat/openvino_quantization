import tensorflow as tf
import numpy as np
import cv2 as cv

from openvino.inference_engine import IECore

class OpenVinoClassification:
    def __init__(self, path_to_xml, path_to_bin): 
        self.ie = IECore()
        self.net = self.ie.read_network(model=path_to_xml, weights=path_to_bin)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify(self, img, top_k):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        image_mean = 127.5
        image_std = 127.5
        image_size_x = 299
        image_size_y = 299 

        img = cv.resize(img, (image_size_x, image_size_y))
        img = np.expand_dims(img, axis=0)
        img = (img - image_mean) / image_std
        img = img.astype(np.float32)
        img = img.transpose((0, 3, 1, 2))

        exec_net = self.ie.load_network(network=self.net, device_name='CPU')
        predictions = exec_net.infer(inputs={input_blob: img})
        predictions = predictions[out_blob]
        predictions = self.sigmoid(predictions)

        labels = np.argsort(predictions[0])[::-1][:5]

        return predictions[0][labels], labels
        