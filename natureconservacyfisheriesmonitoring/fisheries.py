import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import dataset


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

