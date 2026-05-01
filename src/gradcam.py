import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None
		self.activations = None

		# Register hooks
		target_layer.register_forward_hook(self._forward_hook)
		target_layer.register_full_backward_hook(self._backward_hook)

	def _forward_hook(self, module, input, output):
		self.activations = output.detach()

	def _backward_hook(self, module, grad_input, grad_output):
		self.gradients = grad_output[0].detach()

	def generate(self, input_tensor):
		# Forward pass
		output = self.model(input_tensor)

		# Backward pass
		self.model.zero_grad()
		output.backward()

		# Compute Grad-CAM
		gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
		weighted_activations = gradients * self.activations
		heatmap = weighted_activations.mean(dim=1, keepdim=True)[0, 0]

		# Normalize heatmap
		heatmap = F.relu(heatmap)
		heatmap = heatmap / (heatmap.max() + 1e-8)

		return heatmap.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
	"""Overlay heatmap on original image."""
	# Resize heatmap to match image size
	heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

	# Convert image to numpy array
	image_np = np.array(image, dtype=np.float32) / 255.0

	# Apply colormap to heatmap
	heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
	heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
	heatmap_colored = heatmap_colored.astype(np.float32) / 255.0

	# Overlay heatmap on image
	result = (1 - alpha) * image_np + alpha * heatmap_colored
	result = (result * 255).astype(np.uint8)

	return Image.fromarray(result)
