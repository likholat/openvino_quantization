# TensorFlow sample

Link to find ResNet_V2_101 model:
https://www.tensorflow.org/lite/guide/hosted_models
Direct link to download this model:
https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz

To validate TensorFlow model on one image run:
```bash
python3 demo_classification.py --graph resnet_v2_101_299_frozen.pb --image example.jpeg --engine tf
```

To validate TensorFlow model on ImageNet dataset run:
```bash
python3 evaluate.py --graph resnet_v2_101_299_frozen.pb --dataset ILSVRC2012_img_val --engine tf
```
Estimated accuracy:
```bash
Top 1 accuracy: 0.69054
Top 5 accuracy: 0.89814
```

# OpenVINO sample

Run the ```convert.py``` script before converting the model, it creates resnet_v2_101_299_opt.pb file in current directory:
``bash
python3 convert.py --graph resnet_v2_101_299_frozen.pb
``

To convert TensorFlow model to Intermediate Representation:
``bash
python3 /opt/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model resnet_v2_101_299_opt.pb --input_shape "[1,299,299,3]"
``

To initialise OpenVINO environment variables use:
``bash
source /opt/intel/openvino/bin/setupvars.sh
``

To validate OpenVINO model on one image run:
``bash
python3 demo_classification.py --xml resnet_v2_101_299_opt.xml --image example.jpeg --engine opvn
``

To validate OpenVINO model on ImageNet dataset run:
``bash
python3 evaluate.py --xml resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val --engine opvn
``

To quantizes the model to int8 model run:
``bash
python3 calibration.py --xml resnet_v2_101_299_opt.xml --data ILSVRC2012_img_val --annotation ILSVRC2012_img_val/val.txt
``
This script created ```/model/optimised``` folder in current folder with quantized model.

To validate int8 model on one image run:
``bash
python3 demo_classification.py --xml /model/optimized/resnet_v2_101_299_opt.xml --image example.jpeg --engine opvn
``

To validate int8 model on ImageNet dataset run:
``bash
python3 evaluate.py --xml /model/optimized/resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val --engine opvn
``

## Results

Сomparing the benchmark_app results we got an acceleration of about 1.5 times for int8 model:
OpenVINO model benchmark_app result: 
```bash
Latency:    45.98 ms
Throughput: 21.75 FPS
```
int8 model benchmark_app result: 
```bash
Latency:    29.22 ms
Throughput: 34.23 FPS
```

Estimated accuracy for OpenVINO model:
```bash
Top 1 accuracy: 0.69054
Top 5 accuracy: 0.89814
```
Estimated accuracy for int8 model:
```bash
Top 1 accuracy: 0.69316
Top 5 accuracy: 0.89776
```
