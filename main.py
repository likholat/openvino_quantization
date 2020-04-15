import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
from PIL import Image
import numpy as np
import cv2

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def main(argv):
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Import the TF graph
    for arg in sys.argv[1:]:
        print(arg)
        
    graph = load_pb(argv[0])

    labels = []
    # labels_filename = "labels.txt"
    
    # print('out name is ', graph.as_graph_def().node[-1].name)
    # for node in graph.as_graph_def().node:
    #     print(node.op, node.name)
   
    # Load from a file
    imageFile = argv[1]
    image = Image.open(imageFile)

    # Update orientation based on EXIF tags, if the file has orientation info.
    image = convert_to_opencv(image) 

    cv2.imwrite("res.jpg", image)  

    image = resize_down_to_1600_max_dim(image)  

    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)

    augmented_image = resize_to_256_square(max_square_image)

    # with tf.compat.v1.Session() as sess:
    #     input_tensor_shape = sess.graph.get_tensor_by_name('input:0').shape.as_list()
    # network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    # augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    # # These names are part of the model and cannot be changed.
    output_layer = 'output:0'
    input_node = 'input:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            # predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)

    # # Print the highest probability label
    # highest_probability_index = np.argmax(predictions)
    # print('Classified as: ' + labels[highest_probability_index])
    # print()

if __name__ == "__main__":
    main(sys.argv[1:])
