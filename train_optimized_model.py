import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """Load and preprocess the emotion dataset"""
    print("Loading dataset...")
    
    # Load the dataset (assuming it's in the same format as before)
    try:
        df = pd.read_csv('MMAFEDB')
    except:
        print("Dataset not found. Please ensure MMAFEDB file is in the directory.")
        return None, None, None, None
    
    # Extract pixels and emotions
    X = []
    y = []
    
    for index, row in df.iterrows():
        pixels = row['pixels'].split(' ')
        X.append(np.array(pixels, dtype='uint8'))
        y.append(row['emotion'])
    
    X = np.array(X, dtype='uint8')
    y = np.array(y, dtype='uint8')
    
    # Reshape to 48x48x1
    X = X.reshape(X.shape[0], 48, 48, 1)
    
    # Convert to categorical
    y = to_categorical(y, num_classes=7)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def create_optimized_model():
    """Create a lighter, faster model architecture"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the optimized model"""
    print("Starting model training...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    if X_train is None:
        return
    
    # Normalize data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create model
    model = create_optimized_model()
    print("Model architecture:")
    model.summary()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('optimized_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    model_json = model.to_json()
    with open("optimized_model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("Model saved as 'optimized_model.h5' and 'optimized_model.json'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    print("🚀 Starting Optimized Model Training...")
    model = train_model()
    print("✅ Training completed!")
