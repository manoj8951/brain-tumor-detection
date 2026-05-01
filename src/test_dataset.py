from dataset import ImageClassificationDataset


dataset = ImageClassificationDataset("data")

print("Total samples:", len(dataset))

if len(dataset) > 0:
	image, label = dataset[0]
	print("First image shape:", image.shape)
	print("First label:", label)
