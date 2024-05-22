import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Modeli yükle
kullanılacak_model = tf.keras.models.load_model("model.h5")

# Görüntü
image_path = "veriseti/test/Goddess/00000059.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose Estimation
result = pose.process(image)
mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 0, 255), 2, 2))

# Görüntüyü yeniden boyutlandır ve model için uygun hale getir
resized_image = cv2.resize(image, (100, 100))

# Sınıflandırma tahmini
prediction = kullanılacak_model.predict(np.array([resized_image])/255)
index = np.argmax(prediction)
class_names = ["Downdog", "Goddess", "Plank", "Side Plank", "Tree", "Warrior"]
predicted_class = class_names[index]

# Tahmin sonucunu görüntüye yaz
cv2.putText(image, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Görüntüyü göster
cv2.imshow("Pose Estimation", image)

#Beyaz arka plan üzerinde çizim yapmak için
h, w, c = image.shape
opImg = np.zeros([h, w, c])
opImg.fill(255)
mp_draw.draw_landmarks(opImg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 0, 255), 2, 2))
cv2.imshow("Extracted Pose", opImg)

print(result.pose_landmarks)
print(predicted_class)

cv2.waitKey(0)
cv2.destroyAllWindows()
