# Software Requirements 
- TensorFlow 2.5.3
- OpenVINO 2022.1

OpenVINO Development Tools [installation instruction](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html)

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

Run the ```convert.py``` script before converting the model, it creates ```resnet_v2_101_299_opt.pb``` file in current directory:

```bash
python3 convert.py --graph resnet_v2_101_299_frozen.pb
```

To convert TensorFlow model to Intermediate Representation:

```bash
mo --input_model resnet_v2_101_299_opt.pb --input_shape "[1,299,299,3]"
```

To validate OpenVINO model on one image run:

```bash
python3 demo_classification.py --engine opvn --graph resnet_v2_101_299_opt.xml --image example.jpeg 
```

To validate OpenVINO model on ImageNet dataset run:

```bash
python3 evaluate.py --engine opvn --graph resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val
```

To quantizes the model to int8 model run:

```bash
python3 quantize.py --xml resnet_v2_101_299_opt.xml --data ILSVRC2012_img_val --annotation ILSVRC2012_img_val/val.txt
```

This script created ```/model/optimised``` folder in current folder with quantized model.

To validate int8 model on one image run:

```bash
python3 demo_classification.py --engine opvn --graph /model/optimized/resnet_v2_101_299_opt.xml --image example.jpeg 
```

To validate int8 model on ImageNet dataset run:

```bash
python3 evaluate.py --engine opvn --graph /model/optimized/resnet_v2_101_299_opt.xml --dataset ILSVRC2012_img_val
```

# Results

## Estimated accuracy

| Accuracy         | Top 1   | Top 5   |
|:----------------:|:-------:|:-------:|
| TensorFlow model | 0.69054 | 0.89814 |
| OpenVINO model   | 0.69054 | 0.89814 |
| INT8 model       | 0.69088 | 0.89714 |

## Benchmark App results

Сomparing the ```benchmark_app``` results we got an acceleration of about 1.9 times for **INT8 model**:

|                  | Latency (Median)  | Throughput |
|:----------------:|:-----------------:|:----------:|
| OpenVINO model   |     212.16 ms     | 18.02 FPS  |
| INT8 model       |     112.41 ms     | 34.26 FPS  |
