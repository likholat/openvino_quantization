import tensorflow as tf
import argparse
from tensorflow.python.tools import optimize_for_inference_lib

parser = argparse.ArgumentParser()
parser.add_argument('--graph', help='.pb graph path', default='resnet_v2_101_299_frozen.pb')
argv = parser.parse_args()

pb_file = argv.graph
graph_def = tf.compat.v1.GraphDef()

try:
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
except:
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['input'], ['output'], tf.float32.as_datatype_enum)

with tf.gfile.FastGFile('resnet_v2_101_299_opt.pb', 'wb') as f:
   f.write(graph_def.SerializeToString())
   