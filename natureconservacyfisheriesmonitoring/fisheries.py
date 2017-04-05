import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from vis.visualization import visualize_cam
import matplotlib.pyplot as plt

TYPES = ['BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK']

model = load_model('fisheries.h5')

img_path = 'output/train/DOL/img_00165.jpg'
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
print(x.shape)
X = np.expand_dims(x, axis=0)
print(X.shape)

preds = model.predict(X)
pred_class = preds.argmax()
print('Predicted:' + str(preds))
print('Predicted:' + TYPES[pred_class])

heatmap = visualize_cam(model, 5, [pred_class], img)
plt.imshow(heatmap)
