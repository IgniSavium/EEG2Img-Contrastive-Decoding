import os
from PIL import Image
import torch
import numpy as np
from datetime import datetime

current_time = datetime.now().strftime("%m%d_%H%M%S")

# Define the source and target directories
source_dir = f'/path/to/EEG2Img-Contrastive-Decoding/Generation/test_images'
target_dir = f'/path/to/EEG2Img-Contrastive-Decoding/Evaluation/{current_time}/test_images_tensor'

# prev_time = '0815_194901'
# source_dir = f'/path/to/EEG2Img-Contrastive-Decoding/Generation/eeg_image_reconstruction_results/{prev_time}/onlyCLIP-pipeline'
# target_dir = f'/path/to/EEG2Img-Contrastive-Decoding/Evaluation/{current_time}/generated_imgs_tensor'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize a list to hold all the image tensors
tensor_list = []

# Iterate over the folders in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Iterate over the images in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            # Load the image
            with Image.open(image_path) as img:
                # Convert the image to a PyTorch tensor and add a batch dimension
                tensor = torch.tensor(np.array(img)).unsqueeze(0)
                tensor_list.append(tensor)

# Concatenate all tensors along the 0th dimension
all_tensors = torch.cat(tensor_list, dim=0)

# Save the combined tensor
combined_tensor_path = os.path.join(target_dir, "all_images.pt")
torch.save(all_tensors, combined_tensor_path)