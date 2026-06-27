import shutil
from pathlib import Path

def filter_annotated_images():
    labels_dir = Path(r"")
    source_images_dir = Path(r"")
    output_images_dir = Path(r"")

    # Create the output directory 
    output_images_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0

    print("Starting the image extraction process...\n")

    # Iterate through all .txt files in the labels folder
    for txt_path in labels_dir.glob("*.txt"):
        # Skip the classes.txt file if CVAT generated it
        if txt_path.name == "classes.txt":
            continue

        # Extract just the file name without the .txt extension
        base_name = txt_path.stem 

        # Construct the expected path for the corresponding .png image
        source_image_path = source_images_dir / f"{base_name}.png"
        target_image_path = output_images_dir / f"{base_name}.png"

        # Check if the image exists and copy it
        if source_image_path.exists():
            shutil.copy2(source_image_path, target_image_path)
            copied_count += 1
            print(f"Copied: {source_image_path.name}")
        else:
            missing_count += 1
            print(f"WARNING: Image not found for annotation -> {source_image_path.name}")

    # Print a final summary
    print("\n--- Process Complete ---")
    print(f"Successfully copied: {copied_count} images")
    if missing_count > 0:
        print(f"Missing images: {missing_count} (Check the warnings above)")

if __name__ == "__main__":
    filter_annotated_images()