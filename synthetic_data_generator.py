# =============================================================================
# synthetic_data_generator.py - Generate synthetic ASL hand landmarks
# =============================================================================

import numpy as np
import tensorflow as tf
import os
import config

class CVAE(tf.keras.Model):
    """Conditional Variational Autoencoder for synthetic hand landmark generation"""
    
    def __init__(self, latent_dim=32, num_classes=24):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder_inputs = tf.keras.layers.Input(shape=(config.INPUT_SIZE,))
        self.encoder_labels = tf.keras.layers.Input(shape=(num_classes,))
        
        # Combine input with label
        encoder_combined = tf.keras.layers.Concatenate()([
            self.encoder_inputs, 
            self.encoder_labels
        ])
        
        x = tf.keras.layers.Dense(128, activation='relu')(encoder_combined)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Mean and log variance for latent space
        self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = self.sampling([self.z_mean, self.z_log_var])
        
        self.encoder = tf.keras.Model(
            [self.encoder_inputs, self.encoder_labels],
            [self.z_mean, self.z_log_var, z],
            name='encoder'
        )
        
        # Decoder
        self.decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        self.decoder_labels = tf.keras.layers.Input(shape=(num_classes,))
        
        decoder_combined = tf.keras.layers.Concatenate()([
            self.decoder_inputs,
            self.decoder_labels
        ])
        
        x = tf.keras.layers.Dense(64, activation='relu')(decoder_combined)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        decoder_outputs = tf.keras.layers.Dense(config.INPUT_SIZE, activation='tanh')(x)
        
        self.decoder = tf.keras.Model(
            [self.decoder_inputs, self.decoder_labels],
            decoder_outputs,
            name='decoder'
        )
    
    def sampling(self, args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        """Forward pass"""
        x, labels = inputs
        z_mean, z_log_var, z = self.encoder([x, labels])
        reconstructed = self.decoder([z, labels])
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        
        return reconstructed


class SyntheticDataGenerator:
    """Generate synthetic ASL hand landmarks"""
    
    def __init__(self):
        self.cvae = CVAE(latent_dim=32, num_classes=len(config.LETTERS))
        self.cvae.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.MeanSquaredError()
        )
        self.model_path = os.path.join(config.BASE_PATH, 'cvae_generator.h5')
    
    def load_real_data(self):
        """Load existing real data for training the generator"""
        X_data = []
        y_data = []
        
        print("Loading real data for CVAE training...")
        
        for idx, letter in enumerate(config.LETTERS):
            file_path = os.path.join(config.DATA_FOLDER, f"{letter}.npy")
            if os.path.exists(file_path):
                samples = np.load(file_path)
                X_data.extend(samples)
                y_data.extend([idx] * len(samples))
                print(f"  {letter}: {len(samples)} samples")
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Normalize to [-1, 1] for better CVAE training
        X_data = (X_data - 0.5) * 2
        
        # One-hot encode labels
        y_onehot = tf.keras.utils.to_categorical(y_data, len(config.LETTERS))
        
        print(f"Total real samples: {len(X_data)}\n")
        return X_data, y_onehot, y_data
    
    def train_generator(self, epochs=100, batch_size=32):
        """Train the CVAE on real data"""
        X, y_onehot, _ = self.load_real_data()
        
        if len(X) == 0:
            print("No real data found! Collect real data first.")
            return
        
        print("Training CVAE generator...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}\n")
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = self.cvae.fit(
            [X, y_onehot],
            X,  # Reconstruction target
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Save the trained generator
        self.cvae.save_weights(self.model_path)
        print(f"\nCVAE generator saved to: {self.model_path}")
        
        return history
    
    def generate_samples(self, letter, num_samples=1000):
        """Generate synthetic samples for a specific letter"""
        
        # Load trained weights if they exist
        if os.path.exists(self.model_path):
            self.cvae.load_weights(self.model_path)
        else:
            print("Generator not trained! Run train_generator() first.")
            return None
        
        # Get letter index
        if letter not in config.LETTERS:
            print(f"Invalid letter: {letter}")
            return None
        
        letter_idx = config.LETTERS.index(letter)
        
        # Create label vector
        labels = np.zeros((num_samples, len(config.LETTERS)))
        labels[:, letter_idx] = 1
        
        # Sample from latent space
        z = np.random.normal(size=(num_samples, self.cvae.latent_dim))
        
        # Generate synthetic landmarks
        synthetic_data = self.cvae.decoder.predict([z, labels], verbose=0)
        
        # Denormalize from [-1, 1] to [0, 1]
        synthetic_data = (synthetic_data / 2) + 0.5
        
        print(f"Generated {num_samples} synthetic samples for letter '{letter}'")
        return synthetic_data
    
    def augment_dataset(self, samples_per_letter=1000):
        """Generate synthetic data for all letters and save"""
        
        print("\n" + "="*60)
        print("SYNTHETIC DATA AUGMENTATION")
        print("="*60)
        
        for letter in config.LETTERS:
            print(f"\nGenerating synthetic data for '{letter}'...")
            
            # Load existing real data
            real_file = os.path.join(config.DATA_FOLDER, f"{letter}.npy")
            if os.path.exists(real_file):
                real_data = np.load(real_file)
                print(f"  Real samples: {len(real_data)}")
            else:
                real_data = np.array([])
                print(f"  No real data found")
            
            # Generate synthetic data
            synthetic_data = self.generate_samples(letter, samples_per_letter)
            
            if synthetic_data is not None:
                # Combine real and synthetic
                if len(real_data) > 0:
                    combined_data = np.vstack([real_data, synthetic_data])
                else:
                    combined_data = synthetic_data
                
                # Save augmented dataset
                augmented_file = os.path.join(
                    config.DATA_FOLDER, 
                    f"{letter}_augmented.npy"
                )
                np.save(augmented_file, combined_data)
                print(f"  Synthetic samples: {len(synthetic_data)}")
                print(f"  Total samples: {len(combined_data)}")
                print(f"  Saved to: {augmented_file}")
        
        print("\n✅ Synthetic data generation complete!")
        print(f"Use the '*_augmented.npy' files for training")


# =============================================================================
# Usage example
# =============================================================================

def main():
    print("\n" + "="*60)
    print("SYNTHETIC ASL DATA GENERATOR (CVAE)")
    print("="*60)
    print("\nOptions:")
    print("1. Train CVAE generator on real data")
    print("2. Generate synthetic data for all letters")
    print("3. Generate for specific letter")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    generator = SyntheticDataGenerator()
    
    if choice == '1':
        epochs = int(input("Number of epochs (default 100): ") or "100")
        generator.train_generator(epochs=epochs)
    
    elif choice == '2':
        samples = int(input("Samples per letter (default 1000): ") or "1000")
        generator.augment_dataset(samples_per_letter=samples)
    
    elif choice == '3':
        letter = input("Enter letter (A-Y): ").strip().upper()
        samples = int(input("Number of samples (default 1000): ") or "1000")
        synthetic = generator.generate_samples(letter, samples)
        if synthetic is not None:
            save_path = os.path.join(config.DATA_FOLDER, f"{letter}_synthetic.npy")
            np.save(save_path, synthetic)
            print(f"Saved to: {save_path}")
    
    elif choice == '4':
        print("Goodbye!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()