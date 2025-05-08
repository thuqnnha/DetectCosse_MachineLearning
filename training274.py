import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Cấu hình ---
batch_size = 32
img_height = 180
img_width = 180
epochs = 10

# --- Load dữ liệu ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",            # đường dẫn tới thư mục dataset
    validation_split=0.2,  # 80% train, 20% validation
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# --- Tối ưu hiệu suất ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Xây dựng model ---
num_classes = 3  # blue, red, yellow

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)  # Output layer
])

# --- Compile model ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# --- Train model ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- Lưu model sau khi train ---
model.save("model_cos.h5")

# --- Vẽ biểu đồ kết quả ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Độ chính xác train')
plt.plot(epochs_range, val_acc, label='Độ chính xác validation')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Mất mát train')
plt.plot(epochs_range, val_loss, label='Mất mát validation')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
