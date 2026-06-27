import cv2
import numpy as np
from pathlib import Path

def visualize_annotations():
 
    labels_dir = Path(r"")
    images_dir = Path(r"")
    output_dir = Path(r"")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0: Etioplast, 1: PLB, 2: Prothylakoid, 3: Starch Gain, 4: Plastoglobule
    class_colors = {
        0: (0, 0, 255),    # Red
        1: (0, 255, 0),    # Green
        2: (255, 0, 0),    # Blue
        3: (0, 255, 255),  # Yellow (Cyan in BGR)
        4: (255, 0, 255)   # Magenta
    }

    processed_count = 0

    print("Starting visualization process...\n")

    # 3. Iterate through all .txt files
    for txt_path in labels_dir.glob("*.txt"):
        if txt_path.name == "classes.txt":
            continue

        base_name = txt_path.stem
        img_path = images_dir / f"{base_name}.png"
        
        # Check if the matching image exists
        if not img_path.exists():
            print(f"Skipping {base_name}.txt - corresponding image not found.")
            continue

        # 4. Read the image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path.name}")
            continue
            
        height, width = img.shape[:2]

        # 5. Read and draw annotations
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_id = int(parts[0])
            
            # The remaining parts are x,y coordinate pairs
            coords = np.array(parts[1:], dtype=float)
            
            # Reshape into pairs of (x, y)
            points = coords.reshape(-1, 2)
            
            # Denormalize coordinates (multiply x by width, y by height)
            points[:, 0] *= width
            points[:, 1] *= height
            
            # Convert to integers for drawing
            points = np.int32(points)
            # Reshape for cv2.polylines -> (number_of_points, 1, 2)
            points = points.reshape((-1, 1, 2))
            color = class_colors.get(class_id, (255, 255, 255))

            # Draw the polygon outline
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        # 6. Save the visualized image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img)
        processed_count += 1
        print(f"Saved visualization for: {img_path.name}")

    print(f"\n--- Process Complete ---")
    print(f"Successfully visualized and saved {processed_count} images.")

if __name__ == "__main__":
    visualize_annotations()