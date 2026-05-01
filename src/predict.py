import sys

import numpy as np
import torch
from PIL import Image

from model import SimpleCNN


image_path = sys.argv[1]

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

image = Image.open(image_path).convert("RGB")
image = image.resize((128, 128))

image_array = np.array(image, dtype=np.float32) / 255.0
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():
	output = model(image_tensor)
	prediction = 1 if output.item() >= 0.5 else 0

if prediction == 1:
	print("Tumor")
else:
	print("No Tumor")
