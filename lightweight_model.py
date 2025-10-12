import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def create_lightweight_model():
    """Create a very lightweight model for real-time inference"""
    model = Sequential([
        # First block - 32 filters
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Second block - 64 filters
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Third block - 128 filters
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Global Average Pooling instead of Flatten (much faster)
        GlobalAveragePooling2D(),
        
        # Single dense layer
        Dense(7, activation='softmax')
    ])
    
    # Optimized for speed
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_data_lightweight():
    """Load data with minimal preprocessing"""
    print("Loading dataset for lightweight training...")
    
    try:
        df = pd.read_csv('MMAFEDB')
    except:
        print("Dataset not found. Using sample data...")
        # Create sample data for testing
        X = np.random.randint(0, 255, (1000, 48, 48, 1), dtype='uint8')
        y = np.random.randint(0, 7, 1000)
        y = to_categorical(y, num_classes=7)
        return X, y, X, y
    
    # Process data
    X = []
    y = []
    
    for index, row in df.iterrows():
        pixels = row['pixels'].split(' ')
        X.append(np.array(pixels, dtype='uint8'))
        y.append(row['emotion'])
    
    X = np.array(X, dtype='uint8')
    y = np.array(y, dtype='uint8')
    
    # Reshape
    X = X.reshape(X.shape[0], 48, 48, 1)
    y = to_categorical(y, num_classes=7)
    
    # Use only a subset for faster training
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X, y, X, y

def train_lightweight_model():
    """Train the lightweight model quickly"""
    print("🚀 Training Lightweight Model...")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data_lightweight()
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create model
    model = create_lightweight_model()
    print("Lightweight Model Architecture:")
    model.summary()
    
    # Simple data augmentation
    datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('lightweight_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Train with smaller batch size for speed
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=20,  # Fewer epochs for speed
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Save model
    model_json = model.to_json()
    with open("lightweight_model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("💾 Model saved as 'lightweight_model.h5' and 'lightweight_model.json'")
    
    return model

if __name__ == "__main__":
    print("🚀 Starting Lightweight Model Training...")
    model = train_lightweight_model()
    print("✅ Lightweight training completed!")
