# openvino_quantization

Link to find ResNet_V2_101 model:
https://www.tensorflow.org/lite/guide/hosted_models
Direct link to download this model:
https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz

To validate TensorFlow model on one image run:
```bash
python demo_classification_tf.py --graph resnet_v2_101_299_frozen.pb --image example.jpeg
```

To validate TensorFlow model on ImageNet dataset run:
```bash
python3 evaluate_tf.py --graph /resnet_v2_101_299_frozen.pb --dataset /ILSVRC2012_img_val/
```
Estimated accuracy:
```bash

```
