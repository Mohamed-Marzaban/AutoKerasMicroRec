import tensorflow as tf
import autokeras as ak

# Define dataset path (make sure to use 'r' before the path)
dataset_path = r"C:\Users\moham\OneDrive\Desktop\uni\Bachelors\datasets\CASME 2 PREPROCESSED labeled\CASME2 Preprocessed v2"

#Step 1
# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=None,  # Load in batches of 32
    label_mode='int'  # Labels as integers
)


#Step 2
# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
# Apply normalization to the dataset applying it to the image only leaving the labels as they are
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


#Step 3
# Calculate sizes for validation and test datasets
#len gets the number of batches in the dataset
#int() gets the integer value of the number of batches
#val_size is 10% of the dataset for validation
#test_size is 10% of the dataset for testing
# Create validation and test datasets
val_size = int(0.1 * len(train_ds))  # 10% for validation
test_size = int(0.1 * len(train_ds))  # 10% for testing

# Create validation and test datasets
val_ds = train_ds.take(val_size)
test_ds = train_ds.skip(val_size).take(test_size)

# The remaining dataset is used for training
train_ds = train_ds.skip(val_size + test_size)


#step 4 optimizing pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(buffer_size=AUTOTUNE)


#step 5
# Define the AutoKeras ImageClassifier
model = ak.ImageClassifier(
    overwrite=True,  # Start fresh; removes any saved state from previous runs
    max_trials=15 ,   # Try 10 different model architectures
    directory='autokeras_dir'  # Directory to save trial information
)

# Train the model
history = model.fit(
    train_ds,  # Training dataset
    validation_data=val_ds,  # Validation dataset
    epochs=20 , # Number of passes through the entire training dataset
     verbose=2  # Clear logs for each epoch
)

# Export the best model
trained_model = model.export_model()

#  Save the model to disk
# Save as TensorFlow SavedModel
trained_model.save("best_model_tf")




