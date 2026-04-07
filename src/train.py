import tensorflow as tf
from data_generator import VideoDataGenerator
from model import build_model

# PATHS
TRAIN_DIR = r"D:\DataScience\Deep Learning Project\Video_Classifier\data\train\processed\train"
VAL_DIR = r"D:\DataScience\Deep Learning Project\Video_Classifier\data\train\processed\val"

# GENERATORS
train_gen = VideoDataGenerator(TRAIN_DIR, batch_size=8)
val_gen = VideoDataGenerator(VAL_DIR, batch_size=8)

# MODEL
model = build_model()

# CALLBACKS
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_loss',
        save_best_only=True
    )
]

# TRAIN
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# SAVE FINAL MODEL
model.save("final_model.h5")

print("Training completed!")