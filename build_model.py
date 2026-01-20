# =============================================================================
# build_model.py
# =============================================================================

import tensorflow as tf
import config


def create_model():
    """Build the neural network architecture"""

    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(config.INPUT_SIZE,)))

    # Hidden layers
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))

    # Output layer (multi-class classification)
    model.add(
        tf.keras.layers.Dense(
            len(config.LETTERS), activation="softmax"
        )
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LR
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
