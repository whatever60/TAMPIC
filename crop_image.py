import numpy as np


def crop_image(img: np.ndarray, center: tuple[int, int], crop_size: int) -> np.ndarray:
    """
    Crop the image centered at the given coordinates.

    Args:
        img (np.ndarray): Image to be cropped.
        center (tuple[int, int]): Center coordinates for the crop.
        crop_size (int): Size of the crop.

    Returns:
        np.ndarray: Cropped image.
    """
    x, y = center
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    half_crop = crop_size // 2

    # Determine the dimensions of the original image
    height, width = img.shape[:2]

    # Initialize the output image with black boundaries
    if img.ndim == 3:  # For color images
        cropped_image = np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)
    else:  # For grayscale images
        cropped_image = np.zeros((crop_size, crop_size), dtype=img.dtype)

    # Calculate the coordinates to copy from the original image
    copy_y_start = max(0, y - half_crop)
    copy_x_start = max(0, x - half_crop)
    copy_y_end = min(height, y + half_crop)
    copy_x_end = min(width, x + half_crop)

    # Calculate the coordinates to paste into the cropped image
    paste_y_start = max(0, half_crop - y)
    paste_x_start = max(0, half_crop - x)
    paste_y_end = paste_y_start + (copy_y_end - copy_y_start)
    paste_x_end = paste_x_start + (copy_x_end - copy_x_start)

    # Copy the relevant region from the original image to the cropped image
    cropped_image[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = img[
        copy_y_start:copy_y_end, copy_x_start:copy_x_end
    ]

    return cropped_image


if __name__ == "__main__":
    # Sample usage
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Sample image
    center = (50, 50)  # Center coordinates
    crop_size = 60  # Crop size
    cropped_image = crop_image(image, center, crop_size)
