import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_model():
    # Input Shape
    input_shape = (20, 96, 96, 3)

    inputs = layers.Input(shape=input_shape)

    # Pretrained CNN (MobileNetV2)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3)
    )

    base_model.trainable = False  # Freeze initially

    # Apply CNN to each frame
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # Temporal Modeling (GRU)
    x = layers.GRU(64, return_sequences=False)(x)

    # Classification Head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# TEST MODE
if __name__ == "__main__":
    model = build_model()
    model.summary()