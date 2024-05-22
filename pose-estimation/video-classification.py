import cv2
import numpy as np
import tensorflow as tf

# Modeli yükle
kullanılacak_model = tf.keras.models.load_model("model.h5")

# Video akışı al
video_path = "veriseti/train/Side Plank/video18.mp4"  # Video dosya yolu
cap = cv2.VideoCapture(video_path)

class_names = ["Downdog", "Goddess", "Plank", "Side Plank", "Tree", "Warrior"]
class_probabilities = {class_name: 0.0 for class_name in class_names}  # Her sınıf için başlangıçta toplam olasılık sıfır
total_probability = 0.0  # Toplam olasılık başlangıçta sıfır

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü yeniden boyutlandır ve normalleştir
    resized_frame = cv2.resize(frame, (100,100))
    normalized_frame = resized_frame / 255.0

    # Tahmin yap
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

    # Görüntüyü ekrana göster
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video bittiğinde tüm sınıf olasılıklarını konsola yazdır
for class_name, probability in class_probabilities.items():
    # Yüzde değerlerini hesapla ve yazdır
    percent_probability = probability / total_probability * 100
    print(f"{class_name}: {percent_probability:.2f}%")

cap.release()
cv2.destroyAllWindows()
