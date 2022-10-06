import os
import cv2 as cv
import numpy as np
import argparse

from addict import Dict
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import create_pipeline
from openvino.tools.pot import Metric

parser = argparse.ArgumentParser(description="Quantizes OpenVino model to int8.",
                                    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml", default="resnet_v2_101_299_opt.xml", help="XML file for OpenVINO to quantize")
parser.add_argument("--model_name", default="resnet_v2_101_299_opt", help="OpenVINO model name")
parser.add_argument("--annotation", default="val.txt", help="Manifest file (txt file with filenames of images and labels)")
parser.add_argument("--data", default="ILSVRC2012_img_val", help="Data directory root")
parser.add_argument("--int8_dir", default="./model/optimized", help="INT8 directory for calibrated OpenVINO model")

argv = parser.parse_args()

class ImageLoader(DataLoader):
 
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

class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = "Accuracy"
        self._values = []

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        print("{} = {}".format(self.name, sum(self._values)/len(self._values)))

    def update(self, outputs, labels):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """
        result = np.argsort(outputs[0][0])[::-1][:1]

        self._values.append(1 if result[0] == labels[0] else 0)

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {}

# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': argv.model_name,
    "model": argv.xml,
    "weights": argv.xml.split('.xml')[0] + '.bin'
})

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

# Dictionary witn input dataset info
dataset_config = Dict({
    'data_source': argv.data,
    'annotation_file': argv.annotation,
})

# Quantization algorithm settings
algorithms = [
    {
        'name': 'DefaultQuantization', # Optimization algorithm name
        'params': {
            'target_device': 'CPU',
            'preset': 'performance', # Preset [performance (default), accuracy] which controls the quantization mode 
                                     # (symmetric and asymmetric respectively)
            'stat_subset_size': 300  # Size of subset to calculate activations statistics that can be used
                                     # for quantization parameters calculation.
        }
    }
]

# Load the model.
model = load_model(model_config)

# Initialize the data loader and metric.
data_loader = ImageLoader(dataset_config)
metric = AccuracyMetric()

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, metric)

# Initialize the engine for metric calculation and statistics collection.
pipeline = create_pipeline(algorithms, engine)

# Execute the pipeline.
compressed_model = pipeline.run(model)

# Save the compressed model.
save_model(compressed_model, argv.int8_dir)
