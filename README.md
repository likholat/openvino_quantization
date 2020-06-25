# TensorFlow sample

Link to find ResNet_V2_101 model:
https://www.tensorflow.org/lite/guide/hosted_models
Direct link to download this model:
https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz

To validate TensorFlow model on one image run:
```bash
python3 demo_classification.py --engine tf --graph resnet_v2_101_299_frozen.pb --image example.jpeg
```

To validate TensorFlow model on ImageNet dataset run:
```bash
python3 evaluate.py --engine tf --graph resnet_v2_101_299_frozen.pb --dataset ILSVRC2012_img_val
```

# OpenVINO sample

To initialise OpenVINO environment variables use:
``bash
source /opt/intel/openvino/bin/setupvars.sh
``

Run the ```convert.py``` script before converting the model, it creates resnet_v2_101_299_opt.pb file in current directory:
``bash
python3 convert.py --graph resnet_v2_101_299_frozen.pb
``

To convert TensorFlow model to Intermediate Representation:
``bash
cd folder/for/IR/model
python3 ~/openvino/model-optimizer/mo_tf.py --input_shape "[1,299,299,3]" --input_model resnet_v2_101_299_opt.pb 
``

To validate OpenVINO model on one image run:
``bash
python3 demo_classification.py --engine opvn --xml resnet_v2_101_299_opt.xml --image example.jpeg 
``

To validate OpenVINO model on ImageNet dataset run:
``bash
python3 evaluate.py --engine opvn --xml resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val
``

To quantizes the model to int8 model run:
``bash
python3 calibration.py --xml resnet_v2_101_299_opt.xml --data ILSVRC2012_img_val --annotation ILSVRC2012_img_val/val.txt
``
This script created ```/model/optimised``` folder in current folder with quantized model.

To validate int8 model on one image run:
``bash
python3 demo_classification.py --engine opvn --xml /model/optimized/resnet_v2_101_299_opt.xml --image example.jpeg 
``

To validate int8 model on ImageNet dataset run:
``bash
python3 evaluate.py --engine opvn --xml /model/optimized/resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val
``

# Results

## Estimated accuracy

| Accuracy         | Top 1   | Top 5   |
|:----------------:|:-------:|:-------:|
| TensorFlow model | 0.69054 | 0.89814 |
| OpenVINO model   | 0.69054 | 0.89814 |
| INT8 model       | 0.69316 | 0.89776 |

Сomparing the benchmark_app results we got an acceleration of about 1.5 times for int8 model:

## Benchmark App results

|                  | Latency  | Throughput |
|:----------------:|:--------:|:----------:|
| OpenVINO model   | 45.98 ms | 21.75 FPS  |
| INT8 model       | 29.22 ms | 34.23 FPS  |
