import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os


# --- CUSTOM METRICS ---
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# --- LIGHTWEIGHT U-NET FOR LIVE VIDEO ---
def build_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(bn)

    # Decoder
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    u3 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u3)
    c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c6)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c6)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# --- MODEL LOADER ---
def load_flood_model(model_path="flood_best.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = build_unet(input_shape=(128, 128, 3))

        model.load_weights(model_path, by_name=True, skip_mismatch=True)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[precision_m, recall_m, f1_m]
        )

        return model

    except Exception as e:
        raise RuntimeError(f"MODEL LOAD ERROR: {str(e)}")
