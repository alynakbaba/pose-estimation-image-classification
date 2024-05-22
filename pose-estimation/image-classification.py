import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import cv2


kullanılacak_model = tf.keras.models.load_model("model.h5")

image_path = "veriseti/test/Warrior/00000008.jpg"
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

resized_image = cv.resize(image, (100, 100))
prediction = kullanılacak_model.predict(np.array([resized_image])/255)
index = np.argmax(prediction)
class_names = ["Downdog", "Goddess", "Plank", "Side Plank", "Tree", "Warrior"]
predicted_class = class_names[index]
print(predicted_class)
cv.putText(image, predicted_class, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

plt.imshow(image, cmap=plt.cm.binary)
plt.axis('off')
plt.show()
