# import torch
# import matplotlib.pyplot as plt

# # Define a function to load the loss history from a saved model
# def load_loss_history(model_path):
#     checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
#     print(f"Loaded loss history from {model_path}")
#     return checkpoint['losses']

# # List of model paths
# model_paths = ['model/places12/colnet231118-16-33-13-44.pt', 'model/places12.1/colnet231119-14-32-36-44.pt']

# # Plot the loss history for each model
# for model_path in model_paths:
#     loss_history = load_loss_history(model_path)
    
#     # Print loss history for debugging
#     print(f"Loss history for {model_path}: {loss_history}")

#     # Plot training loss
#     print("Train")
#     plt.plot(loss_history['train'], label=f'Train - {model_path}')
#     print(f"Loss history for {model_path}: {loss_history['train']}")
#     print("Val")
#     # Plot validation loss
#     plt.plot(loss_history['val'], label=f'Validation - {model_path}')
#     print(f"Loss history for {model_path}: {loss_history['val']}")


# # Add labels and legend
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Show the plot
# plt.show()

# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# # Define a function to load the loss history from a saved model
# def load_loss_history(model_path):
#     checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
#     print(f"Loaded loss history from {model_path}")
#     return checkpoint['losses']

# # List of model paths
# model_paths = ['model/places12/colnet231118-16-33-13-44.pt', 'model/places12.1/colnet231119-14-32-36-44.pt']

# # Plot the loss history for each model
# for model_path in model_paths:
#     loss_history = load_loss_history(model_path)
    
#     # Print loss history for debugging
#     print(f"Loss history for {model_path}: {loss_history}")

#     # Plot training loss
#     plt.plot(np.log(loss_history['train']), label=f'Train - {model_path}')
    
#     # Plot validation loss
#     plt.plot(np.log(loss_history['val']), label=f'Validation - {model_path}')

# # Add labels and legend
# plt.xlabel('Epoch')
# plt.ylabel('Log Loss')
# plt.legend()

# # Set y-axis scale to logarithmic
# plt.yscale('log')

# # Show the plot
# plt.show()

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define a function to load the loss history from a saved model
def load_loss_history(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    print(f"Loaded loss history from {model_path}")
    return checkpoint['losses']

# List of model paths
model_paths = ['model/places12/colnet231118-16-33-13-44.pt',
              'model/places12.1/colnet231119-14-32-36-44.pt',
              'model/places12.2/colnet231122-14-02-03-44.pt',
              'model/places12.3/colnet231123-10-00-22-44.pt']
# Increase the size of the plots
# plt.figure(figsize=(10, 6))
save_path = 'plot_dl/loss'
# Plot the loss history for each model
for i, model_path in enumerate (model_paths):
    loss_history = load_loss_history(model_path)
    name = f"places12.{i}.png"
    # Print loss history for debugging
    print(f"Loss history for {model_path}: {loss_history}")

    # Create a new plot for each model with a larger size
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(loss_history['train'], label='Train')
    
    # Plot validation loss
    plt.plot(loss_history['val'], label='Validation')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Colorization + Classification Loss')
    plt.title(f'Loss History for Colnet {i}')
    plt.legend()

    print("Saved fig")

    plt.savefig(os.path.join(save_path, name))

# Show the plots
plt.show()

