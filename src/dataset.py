import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageClassificationDataset(Dataset):
	def __init__(self, data_dir="data", augment=False):
		self.data_dir = data_dir
		self.classes = {"no": 0, "yes": 1}
		self.samples = []
		self.transform = self._build_transform(augment)

		for class_name, label in self.classes.items():
			class_dir = os.path.join(data_dir, class_name)
			if not os.path.isdir(class_dir):
				continue

			for file_name in sorted(os.listdir(class_dir)):
				file_path = os.path.join(class_dir, file_name)
				if os.path.isfile(file_path):
					self.samples.append((file_path, label))

	def _build_transform(self, augment):
		if augment:
			return transforms.Compose([
				transforms.Resize((128, 128)),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(15),
				transforms.ToTensor(),
			])

		return transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		file_path, label = self.samples[index]
		image = Image.open(file_path).convert("RGB")
		image_tensor = self.transform(image)

		return image_tensor, label
