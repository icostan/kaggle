import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import dataset
from matplotlib.patches import Rectangle
import json
import cv2
import os
import skimage as ski
from PIL import Image


def visualize():
    model = load_model('fisheries.h5')

    path = 'input/test_stg1/img_00009.jpg'
    img = utils.load_img(path, target_size=(dataset.SIZE, dataset.SIZE))
    X = dataset.load_image(path)
    print(X.shape)
    # print(X[0])

    preds = model.predict(X)
    pred_class = preds.argmax()
    print('Predicted:' + str(preds))
    print('Predicted:' + dataset.TYPES[pred_class])

    plt.imshow(img)
    plt.show()

    idx = [2, 6, 8, 13]
    for i in idx:
        print(model.layers[i])
        heatmap = visualize_cam(model, i, [pred_class], img)
        plt.imshow(heatmap)
        plt.show()


def show_annotated_images():
    fish = 'DOL'
    fish_path = 'input/train/{fish}/'.format(fish=fish)

    files = os.listdir(fish_path)
    for filename in files:
        img = Image.open(fish_path + filename)
        a = annotation(filename, fish)

        x = a['x']
        y = a['y']
        h = a['height']
        w = a['width']

        fish_img = img.crop((x, y, x + w, y + h))
        plt.imshow(fish_img)

        # plt.gca().add_patch(
        #     Rectangle((a['x'], a['y']), a['width'], a['height'], fill=None))

        plt.show()


def annotation(filename, fish):
    for a in annotations(fish):
        if a['filename'] == filename:
            return a['annotations'][0]


def annotations(fish='lag'):
    fish = fish.lower()
    path = 'annotations/{fish}_labels.json'.format(fish=fish)
    with open(path) as data_file:
        annotations = json.load(data_file)
    return annotations


show_annotated_images()
