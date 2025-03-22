import tensorflow as tf

# Define dataset path (make sure to use 'r' before the path)
dataset_path = r"C:\Users\moham\OneDrive\Desktop\uni\Bachelors\datasets\CASME 2 PREPROCESSED labeled\CASME2 Preprocessed v2"

#Step 1
# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=32,  # Load in batches of 32
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


print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Test batches: {len(test_ds)}")





