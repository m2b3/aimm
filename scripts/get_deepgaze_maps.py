import argparse
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
import deepgaze_pytorch
import torchvision.transforms as transforms
import os

# Dataset selection - the default matches the paper's main dataset
parser = argparse.ArgumentParser(description="Generate DeepGaze IIE maps for one dataset.")
parser.add_argument("--dataset", default="P21_scegram",
                    help='e.g. "P21_scegram", "HH25_indoor", "HH25_outdoor", "TEST"')
DATASET = parser.parse_args().dataset

DEVICE = 'cpu'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

# Configure paths based on dataset - all paths follow same structure with dataset name
inpath = f'data/{DATASET}/images'
outpath = f'data/{DATASET}/maps/deepgaze_ncb' # select _cb or _ncb

# With center bias?
# centerbias_template = np.load('data/templates_cb/centerbias_mit1003.npy') # _cb, # precomputed centerbias log density (from MIT1003) over a 1024x1024 image
centerbias_template = np.zeros((1024, 1024)) # _ncb

# Create output directory if it doesn't exist
os.makedirs(outpath, exist_ok=True)



###############
# Run model
###############

# Process all images in the input folder
for filename in os.listdir(inpath):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Load image
            image_path = os.path.join(inpath, filename)
            image = plt.imread(image_path)

            # Convert PNG to JPG format if needed
            if filename.lower().endswith('.png'):
                # Remove alpha channel if present
                if len(image.shape) == 3 and image.shape[-1] == 4:
                    image = image[:, :, :3]
                # Convert float [0,1] to uint8 [0,255] if needed
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
            
            # rescale to match image size
            centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
            # renormalize log density
            centerbias -= logsumexp(centerbias)

            image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
            centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

            log_density_prediction = model(image_tensor, centerbias_tensor)

            # Squeeze to remove dimensions of size 1
            image_tensor = log_density_prediction.squeeze()


            # Normalize the tensor to a 0-255 range for image representation
            min_val = image_tensor.min()
            max_val = image_tensor.max()
            normalized_tensor = (image_tensor - min_val) / (max_val - min_val)
            image_tensor_255 = (normalized_tensor * 255).type(torch.uint8)

            # Convert to PIL image
            output_image = transforms.ToPILImage()(image_tensor_255)

            # Save with same filename in output folder
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(outpath, f"{name}.png")
            output_image.save(output_path)

            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Batch processing complete!")