import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def display_images_with_same_name(image_name, original_path, additional_paths, labels, label):
    """
    Display images with the same name from different directories.

    Parameters:
    - image_name: The filename of the image to display.
    - original_path: The path to the directory containing the original image.
    - additional_paths: A list of paths to directories containing additional images with the same name.
    - labels: A list of labels for the corresponding additional images.
    """
    fig, axs = plt.subplots(1, len(additional_paths) + 1, figsize=(15, 4))

    # Load and display the original image
    original_image_path = os.path.join(original_path, image_name)
    original_image = Image.open(original_image_path)
    axs[0].imshow(original_image)
    axs[0].set_title('Original\n{}'.format(labels[0]))
    axs[0].axis('off')

    # Display additional images
    for i, (additional_path, label) in enumerate(zip(additional_paths, labels[1:])):
        additional_image_path = os.path.join(additional_path, label + image_name)
        additional_image = Image.open(additional_image_path)
        axs[i + 1].imshow(additional_image)
        axs[i + 1].set_title(label)
        axs[i + 1].axis('off')
    # plt.show()
    plt.savefig(fname = '/comparisons/' + image_name)

# Example usage:
image_name = '00002037.jpg'
label = 'dorm_room'
original_path = './data/places12/test/' + label
additional_paths = ['./out/places12', './out/places12.1']
labels = ['Original', 'Colnet', 'Modified Colnet']

display_images_with_same_name(image_name, original_path, additional_paths, labels, label)
