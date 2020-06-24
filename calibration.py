import os
import cv2 as cv
import numpy as np
import argparse

from addict import Dict
from compression.graph import load_model, save_model
from compression.data_loaders.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.custom.metric import Metric
from compression.pipeline.initializer import create_pipeline

parser = argparse.ArgumentParser(description="Quantizes OpenVino model to int8.",
                                    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml", default="resnet_v2_101_299_opt.xml", help="XML file for OpenVINO to quantize")
parser.add_argument("--bin", default="resnet_v2_101_299_opt.bin", help="BIN file for OpenVINO to quantize")
parser.add_argument("--model_name", default="resnet_v2_101_299_opt", help="OpenVINO model name")
parser.add_argument("--annotation", default="val.txt", help="Manifest file (txt file with filenames of images and labels)")
parser.add_argument("--data", default="ILSVRC2012_img_val", help="Data directory root")
parser.add_argument("--int8_dir", default="./model/optimized", help="INT8 directory for calibrated OpenVINO model")

argv = parser.parse_args()

class MyDataLoader(DataLoader):
 
    def __init__(self, config):
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)

        imagesIds = {}

        with open(config['annotation_file'], 'rt') as f:
             dataset = f.read().strip().split('\n') 

        self.images = []
        self.annotations = []

        for i, value in enumerate(dataset):
            img_path, label = value.rsplit(' ')
            label = int(label) + 1

            img_path = os.path.join(config['data_source'], img_path)

            self.images.append(img_path)
            self.annotations.append(label)

        self.images = self.images[:1000]

    @property
    def size(self):
        return len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        img = cv.imread(self.images[item])

        image_mean = 127.5
        image_std = 127.5
        image_size_x = 299
        image_size_y = 299 

        img = cv.resize(img, (image_size_x, image_size_y))
        img = np.expand_dims(img, axis=0)
        img = (img - image_mean) / image_std
        img = img.astype(np.float32)
        img = img.transpose((0, 3, 1, 2))

        return (item, self.annotations[item]), img

class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = "Accuracy"
        self._values = []
        self.round = 1

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        print(len(self._values))    #print res: 300
        print("Round #{}  {} = {}".format(self.round, self.name, sum(self._values)/len(self._values)))
        self.round += 1

    def update(self, outputs, labels):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """
        result = np.argsort(outputs[0][0])[::-1][:5]

        if (result[0] == labels[0]):
            self._values.append(1)
        else:
            self._values.append(0)

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {self.name: {"direction": "higher-better", "type": ""}}

model_config = Dict({
    'model_name': argv.model_name,
    "model": argv.xml,
    "weights": argv.bin
})
engine_config = {
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
}
dataset_config = {
    'data_source': argv.data,
    'annotation_file': argv.annotation,
}
algorithms = [
    {
        'name': 'DefaultQuantization',
        'params': {
            'target_device': 'CPU',
            'preset': 'performance',
            'stat_subset_size': 300
        }
    }
]

model = load_model(model_config)

data_loader = MyDataLoader(dataset_config)
metric = MyMetric()

loss = None
engine = IEEngine(engine_config, data_loader, metric, loss)
pipeline = create_pipeline(algorithms, engine)

compressed_model = pipeline.run(model)
save_model(compressed_model, argv.int8_dir)
