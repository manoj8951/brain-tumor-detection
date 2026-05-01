import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from dataset import ImageClassificationDataset


train_dataset = ImageClassificationDataset("data/train", augment=True)
test_dataset = ImageClassificationDataset("data/test", augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)

for parameter in model.parameters():
	parameter.requires_grad = False

for parameter in model.layer3.parameters():
	parameter.requires_grad = True

for parameter in model.layer4.parameters():
	parameter.requires_grad = True

for parameter in model.fc.parameters():
	parameter.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
	list(model.layer3.parameters()) + list(model.layer4.parameters()) + list(model.fc.parameters()),
	lr=0.001,
)
best_accuracy = 0

for epoch in range(10):
	total_loss = 0.0

	for images, labels in train_loader:
		labels = labels.float().unsqueeze(1)

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

	correct = 0
	total = 0
	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for images, labels in test_loader:
			outputs = model(images)
			predictions = (torch.sigmoid(outputs) >= 0.4).float().squeeze(1)
			correct += (predictions == labels.float()).sum().item()
			total += labels.size(0)
			all_labels.extend(labels.tolist())
			all_predictions.extend(predictions.tolist())

	accuracy = (correct / total) * 100 if total > 0 else 0
	print(f"Test Accuracy: {accuracy:.2f}%")

	cm = confusion_matrix(all_labels, all_predictions)
	precision = precision_score(all_labels, all_predictions, zero_division=0)
	recall = recall_score(all_labels, all_predictions, zero_division=0)
	f1 = f1_score(all_labels, all_predictions, zero_division=0)

	print("Confusion Matrix:")
	print(cm)
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1-Score: {f1:.4f}")

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		torch.save(model.state_dict(), "best_model.pth")
		print("Saved best_model.pth")
