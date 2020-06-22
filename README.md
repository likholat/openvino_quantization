# openvino_quantization

Link to find ResNet_V2_101 model:
https://www.tensorflow.org/lite/guide/hosted_models
Direct link to download this model:
https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz

To validate TensorFlow model on one image run:
```bash
python demo_classification_tf.py --graph /path/to/resnet_v2_101_299_frozen.pb --image /path/to/example.jpeg
```

To validate TensorFlow model on ImageNet dataset run:
```bash
python3 evaluate_tf.py --graph /path/to/resnet_v2_101_299_frozen.pb --dataset /path/to/ILSVRC2012_img_val
```
Estimated accuracy:
```bash
Top 1 accuracy: 0.69054
Top 5 accuracy: 0.89814
```

# openvino_quantization

To initialise OpenVINO environment variables open the Command Prompt, and run the setupvars.bat batch file:
``bash
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
``

To validate OpenVINO model on one image run:
``bash
python demo_classification_opvn.py --graph /path/to/resnet_converted --image /path/to/example.jpeg
``

To validate OpenVINO model on ImageNet dataset run:
``bash
python evaluate_opvn.py --graph /path/to/resnet_converted --dataset /path/to/ILSVRC2012_img_val
``