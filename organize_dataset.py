import os
import shutil
import zipfile

zip_file = "brain-tumor-mri-dataset.zip"
extract_path = "extracted"
final_dataset_path = "dataset"

# Step 1: Unzip
print("🔄 Unzipping dataset...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 2: Define training folders
training_path = os.path.join(extract_path, "Training")

tumor_folders = ["glioma", "meningioma", "pituitary"]
no_tumor_folder = "notumor"

# Step 3: Create target folders
yes_path = os.path.join(final_dataset_path, "yes")
no_path = os.path.join(final_dataset_path, "no")
os.makedirs(yes_path, exist_ok=True)
os.makedirs(no_path, exist_ok=True)

# Step 4: Copy tumor images
for tumor_type in tumor_folders:
    folder_path = os.path.join(training_path, tumor_type)
    print(f"📁 Organizing tumor images from: {tumor_type}")
    for img in os.listdir(folder_path):
        src = os.path.join(folder_path, img)
        dst = os.path.join(yes_path, f"{tumor_type}_{img}")
        shutil.copy(src, dst)

# Step 5: Copy non-tumor images
no_tumor_path = os.path.join(training_path, no_tumor_folder)
print("📁 Organizing no tumor images...")
for img in os.listdir(no_tumor_path):
    src = os.path.join(no_tumor_path, img)
    dst = os.path.join(no_path, img)
    shutil.copy(src, dst)

print("✅ Dataset organized into 'dataset/yes' and 'dataset/no'. Ready for training!")
