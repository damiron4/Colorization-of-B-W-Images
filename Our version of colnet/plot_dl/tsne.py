import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load your model
# model = ...

# Load your data
# data_loader = ...

# Set the model to evaluation mode
model.eval()

# Collect feature vectors and labels
feature_vectors = []
labels = []

with torch.no_grad():
    for inputs, targets in data_loader:
        # Forward pass up to the layer you want to extract features from
        outputs = model.extract_features(inputs)  # Modify this based on your model architecture

        # Collect feature vectors and labels
        feature_vectors.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())

# Concatenate the collected feature vectors and labels
feature_vectors = np.concatenate(feature_vectors, axis=0)
labels = np.concatenate(labels, axis=0)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(feature_vectors)

# Plot the t-SNE embeddings
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()
