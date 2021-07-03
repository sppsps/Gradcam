import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
vgg_model = VGG16()
import tensorflow as tf
import keras
import numpy as np
import cv2
model = keras.models.Model(inputs = vgg_model.input, outputs = [vgg_model.get_layer(index = 17).output,vgg_model.output])
def get_gradients(input, model, pred_index = None):
    input = np.expand_dims(input, axis = 0)
    image = tf.cast(input, tf.float32)
    preds = vgg_model
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = model(image)
        # print(preds)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        # print(class_channel)
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        w = image.shape[2]
        h = image.shape[1]
        heatmap = cv2.resize(heatmap.numpy(), (w, h))
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        #heatmap = tf.image.resize(heatmap, (image[0],image[1]))
        return heatmap.numpy()

import cv2
from PIL import Image
import numpy as np
import argparse
import requests
import urllib.request
parser = argparse.ArgumentParser()
parser.add_argument('--img_src', 
    default = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/35-1315924315-g.jpg/149px-35-1315924315-g.jpg", help = 'img_src')
args, unknown = parser.parse_known_args()
urllib.request.urlretrieve(args.img_src, 'image')
#im = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/35-1315924315-g.jpg/149px-35-1315924315-g.jpg", stream=True).raw)
im = Image.open("image")
im = im.resize((224,224))
plt.imshow(im)

heatmap = get_gradients(im, model, None)

plt.matshow(heatmap)
plt.show()

