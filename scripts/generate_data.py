import os
import random
from PIL import Image, ImageDraw


def create_shape_image(shape_type, size=(32, 32)):
    # Create a blank black image
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)

    # Generate random coordinates for the shape
    padding = 4
    x1 = random.randint(padding, size[0] // 2 - padding)
    y1 = random.randint(padding, size[1] // 2 - padding)
    x2 = random.randint(size[0] // 2 + padding, size[0] - padding)
    y2 = random.randint(size[1] // 2 + padding, size[1] - padding)

    # Generate random color
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    if shape_type == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)
    else:  # rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def generate_dataset(base_dir, num_train=100, num_val=20):
    splits = {"train": num_train, "val": num_val}
    shapes = ["circles", "rectangles"]

    for split, num_samples in splits.items():
        for shape in shapes:
            dir_path = os.path.join(base_dir, split, shape)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(num_samples):
                img = create_shape_image(shape[:-1])  # remove 's' for shape type
                file_path = os.path.join(dir_path, f"{i:03d}.png")
                img.save(file_path)

    print(f"Dataset generated at {base_dir}")
    print(f"Train: {num_train} samples per class")
    print(f"Val: {num_val} samples per class")


if __name__ == "__main__":
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "shapes"
    )
    generate_dataset(base_dir)
