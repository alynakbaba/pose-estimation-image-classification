from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_dir = 'veriseti/train'
test_dir = 'veriseti/test'

# Veri artırma ve ön işleme
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')

# Model oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Model derleme
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
epochs = 20
model.fit(
      train_generator,
      epochs=epochs,
      validation_data=None,
      verbose=1)

model.save("model.h5")