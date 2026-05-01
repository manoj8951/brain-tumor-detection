import os
import random
import shutil


data_dir = "data"
classes = ["yes", "no"]
split_ratio = 0.8


for split in ["train", "test"]:
	for class_name in classes:
		os.makedirs(os.path.join(data_dir, split, class_name), exist_ok=True)


for class_name in classes:
	source_dir = os.path.join(data_dir, class_name)
	files = [file_name for file_name in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file_name))]
	random.shuffle(files)

	split_index = int(len(files) * split_ratio)
	train_files = files[:split_index]
	test_files = files[split_index:]

	for file_name in train_files:
		source_path = os.path.join(source_dir, file_name)
		target_path = os.path.join(data_dir, "train", class_name, file_name)
		shutil.copy2(source_path, target_path)

	for file_name in test_files:
		source_path = os.path.join(source_dir, file_name)
		target_path = os.path.join(data_dir, "test", class_name, file_name)
		shutil.copy2(source_path, target_path)


print("Dataset split complete.")