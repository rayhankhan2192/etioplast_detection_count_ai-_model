import os
import random
import shutil

# Source paths (Train directories)
train_images_dir = r""
train_labels_dir = r""

# Destination paths (Val directories)
val_images_dir = r""
val_labels_dir = r""

val_split = 0.20

# Create the Val directories if they don't already exist
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get all images in the Train directory (filtering for standard image extensions)
valid_extensions = ('.jpg', '.jpeg', '.png')
all_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(valid_extensions)]

# Calculate the exact number of files to move
num_val_files = int(len(all_images) * val_split)

# Randomly sample the images for an unbiased validation set
val_images_selected = random.sample(all_images, num_val_files)

moved_count = 0
missing_labels = []

for img_name in val_images_selected:
    # 1. Define image paths
    src_image_path = os.path.join(train_images_dir, img_name)
    dst_image_path = os.path.join(val_images_dir, img_name)
    
    # 2. Extract base name to find the exact matching .txt file
    base_name = os.path.splitext(img_name)[0]
    label_name = base_name + '.txt'
    
    src_label_path = os.path.join(train_labels_dir, label_name)
    dst_label_path = os.path.join(val_labels_dir, label_name)
    
    # 3. Check if the matching label file exists before moving anything
    if os.path.exists(src_label_path):
        # Cut and paste (move) the image and the label
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_label_path, dst_label_path)
        moved_count += 1
    else:
        # Track if any text files are missing so you can debug later if needed
        missing_labels.append(label_name)

print(f"Task Complete! Successfully moved {moved_count} image-label pairs to the Val folders.")
print(f"Remaining in Train: {len(all_images) - moved_count} pairs.")

if missing_labels:
    print(f"\nWarning: Could not find label files for {len(missing_labels)} images. These images were skipped:")
    for missing in missing_labels[:5]: # Show first 5 missing for brevity
        print(f" - {missing}")