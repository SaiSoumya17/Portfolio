import numpy as np
from sklearn.model_selection import train_test_split

# Load combined data and labels
X = np.load('X_combined.npy')
y = np.load('y_combined.npy')

# Step 1: Split into train + validation and test sets (e.g., 80% for train+validation, 20% for test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Further split train + validation into train and validation sets (e.g., 80% for train, 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

# Optionally, save the split datasets to separate .npy files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Reshape the labels to remove the unnecessary dimension
y_train = y_train.reshape(-1, 2)  # Now y_train will have shape (1588, 2)
y_val = y_val.reshape(-1, 2)
y_test = y_test.reshape(-1, 2)

# Now you can check the shape of your labels
print(f"Reshaped Train labels shape: {y_train.shape}")
print(f"Reshaped Validation labels shape: {y_val.shape}")
print(f"Reshaped Test labels shape: {y_test.shape}")

