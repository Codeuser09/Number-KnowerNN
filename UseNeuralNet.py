import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('digits.model')

for x in range(1, 6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))

    prediction = model.predict(img)
    print(f'It is a {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
