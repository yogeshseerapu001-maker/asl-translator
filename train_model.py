# =============================================================================
# train_model.py
# =============================================================================

import numpy as np
import os
from sklearn.model_selection import train_test_split
from build_model import create_model
import tensorflow as tf  # ADD THIS LINE
import config

def load_training_data():
    """Load all the samples we collected"""
    
    X_data = []
    y_data = []
    
    print("Loading data from files...")
    
    for idx, letter in enumerate(config.LETTERS):
        file_path = os.path.join(config.DATA_FOLDER, f"{letter}.npy")
        
        if os.path.exists(file_path):
            samples = np.load(file_path)
            X_data.extend(samples)
            y_data.extend([idx] * len(samples))
            print(f"  {letter}: {len(samples)} samples")
        else:
            print(f"  {letter}: NO DATA FOUND")
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\nTotal samples loaded: {len(X_data)}")
    return X_data, y_data


def train():
    """Main training function"""
    
    # load the data
    X, y = load_training_data()
    
    if len(X) == 0:
        print("\nNo training data found!")
        print("Run the data collection script first.")
        return
    
    # split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.VAL_SPLIT,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # build the model
    print("\nBuilding model...")
    model = create_model()
    model.summary()
    
    # setup callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(  # NOW THIS WORKS
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(  # NOW THIS WORKS
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\nStarting training...\n")
    
    # train!
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nTraining complete!")
    print(f"Final validation accuracy: {val_acc*100:.2f}%")
    
    # save the model
    model.save(config.MODEL_FILE)
    print(f"Model saved to: {config.MODEL_FILE}")