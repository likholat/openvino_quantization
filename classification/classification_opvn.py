import tensorflow as tf
import numpy as np
import cv2 as cv

from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

class OpenVinoClassification:
    def __init__(self, path_to_xml, path_to_bin): 
        self.core = Core()
        net = self.core.read_model(path_to_xml, path_to_bin)

        ppp = PrePostProcessor(net)
        ppp.input().tensor().set_layout(Layout('NCHW'))
        ppp.input().model().set_layout(Layout('NHWC'))

        net = ppp.build()
        self.compiled_model = self.core.compile_model(net, 'CPU')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify(self, img, top_k):
        image_mean = 127.5
        image_std = 127.5
        image_size_x = 299
        image_size_y = 299

        img = cv.resize(img, (image_size_x, image_size_y))
        img = np.expand_dims(img, axis=0)
        img = (img - image_mean) / image_std
        img = img.astype(np.float32)
        img = img.transpose((0, 3, 1, 2))

        results = self.compiled_model.infer_new_request({0: img})

        predictions = next(iter(results.values()))
        predictions = self.sigmoid(predictions)

        labels = np.argsort(predictions[0])[::-1][:5]

        return predictions[0][labels], labels
        