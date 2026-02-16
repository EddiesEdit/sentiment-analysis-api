print("Loading app/model.py...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, Conv1D, Embedding, Dropout
from tensorflow.keras.optimizers import Adam


def build_model_v1(vocab_size, embed_size=300, input_length=1000):
    """
    First CNN model (basic version)
    """
    model = Sequential([
        Embedding(vocab_size, embed_size, input_length=input_length),
        Conv1D(128, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(32, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_v2(vocab_size, embed_size=300, input_length=1000, learning_rate=1e-4):
    """
    Second CNN model (with learning rate adjustment)
    """
    model = Sequential([
        Embedding(vocab_size, embed_size, input_length=input_length),
        Conv1D(128, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(32, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def build_model_v3(vocab_size, embed_size=300, input_length=1000, learning_rate=1e-4, dropout_rate=0.3):
    """
    Third CNN model (with dropout and regularization)
    """
    model = Sequential([
        Embedding(vocab_size, embed_size, input_length=input_length),

        Conv1D(128, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Dropout(dropout_rate),

        Conv1D(64, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Dropout(dropout_rate),

        Conv1D(32, 4, padding='same', activation='relu'),
        MaxPooling1D(2),
        Dropout(dropout_rate),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
print("Finished loading app/model.py.")