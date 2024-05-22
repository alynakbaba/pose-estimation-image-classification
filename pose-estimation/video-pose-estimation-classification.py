import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Modeli yükle
kullanılacak_model = tf.keras.models.load_model("model.h5")

# Video
video_path = "veriseti/train/Warrior/video22.mp4"
cap = cv2.VideoCapture(video_path)
## Kamera
# cap = cv2.VideoCapture(0)

# FPS değeri al
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
sample_rate = int(fps / 5)  # Örnek alma sıklığını ayarla

current_pose = None
pose_count = 0
threshold_pose_count = 15  # Bu pozda geçiş için eşik değeri

class_names = ["Downdog", "Goddess", "Plank", "Side Plank", "Tree", "Warrior"]
class_probabilities = {class_name: 0.0 for class_name in class_names}
total_probability = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS'e göre örnek alma sıklığını kontrol et
    if frame_count % sample_rate != 0:
        continue

    # Görüntüyü yeniden boyutlandır ve normalleştir
    resized_frame = cv2.resize(frame, (100, 100))
    normalized_frame = resized_frame / 255.0

    # Pose Estimation
    result = pose.process(frame)
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2))

    # Sınıflandırma tahmini
    prediction = kullanılacak_model.predict(np.expand_dims(normalized_frame, axis=0))[0]
    max_index = np.argmax(prediction)
    max_probability = np.max(prediction)
    predicted_class = class_names[max_index]
    text = f"{predicted_class}: {max_probability * 100:.2f}%\n"
    print(text)

    # Tahmin sonucunu görüntüye yaz
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Her sınıf için olasılığı güncelle
    class_probabilities[predicted_class] += max_probability

    # Toplam olasılığı güncelle
    total_probability += max_probability

    if predicted_class != current_pose or pose_count >= threshold_pose_count:
        # Şu anki pozun geçişini yazdır
        print(f"Geçiş: {current_pose} --> {predicted_class}")
        current_pose = predicted_class
        pose_count = 0  # Poz sayacını sıfırla
    else:
        pose_count += 1  # Poz sayacını artır

    # Görüntüyü ekrana göster
    cv2.imshow("Pose Estimation", frame)

    # Beyaz arka plan üzerinde çizim yapmak için
    h, w, c = frame.shape
    opImg = np.zeros([h, w, c])
    opImg.fill(255)
    mp_draw.draw_landmarks(opImg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2))
    cv2.imshow("Extracted Pose", opImg)

    print(result.pose_landmarks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video bittiğinde tüm sınıf olasılıklarını yazdır
for class_name, probability in class_probabilities.items():
    # Yüzde değerlerini hesapla ve yazdır
    percent_probability = probability / total_probability * 100
    print(f"{class_name}: {percent_probability:.2f}%")

cap.release()
cv2.destroyAllWindows()
