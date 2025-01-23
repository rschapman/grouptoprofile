import cv2
import numpy as np
import os

def circle_crop_face(img, x, y, w, h, min_dim=256, margin_factor=1.2):
    """
    Crop the face region from the image, expand bounding box by a margin,
    and turn it into a circle with a white background.

    :param img:           Original BGR image
    :param x, y, w, h:    Face bounding box coordinates
    :param min_dim:       Minimum dimension of the output circle
    :param margin_factor: How much to expand the bounding box. 1.2 = 20% bigger
    """
    # Calculate center of the face box
    cx = x + w // 2
    cy = y + h // 2

    # Expand bounding box by margin_factor
    new_w = int(w * margin_factor)
    new_h = int(h * margin_factor)

    # Adjust so that the new ROI is centered the same
    new_x = cx - new_w // 2
    new_y = cy - new_h // 2

    # Clamp coordinates so we don't go outside the image
    # (necessary if the face is near the edges)
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_w > img.shape[1]:
        new_w = img.shape[1] - new_x
    if new_y + new_h > img.shape[0]:
        new_h = img.shape[0] - new_y

    # Extract the (potentially larger) face ROI
    face_roi = img[new_y:new_y+new_h, new_x:new_x+new_w]

    # Use the larger dimension to form a square
    dim = max(new_w, new_h)

    # Enforce a minimum dimension (e.g. 256)
    dim = max(dim, min_dim)

    # Resize ROI to a square
    square_face = cv2.resize(face_roi, (dim, dim), interpolation=cv2.INTER_AREA)

    # Create a mask with a white circle
    mask = np.zeros((dim, dim, 3), dtype=np.uint8)
    cv2.circle(mask, (dim//2, dim//2), dim//2, (255, 255, 255), -1)

    # Convert the circle mask to a single-channel binary mask
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    # Create an all-white background
    white_bg = np.ones((dim, dim, 3), dtype=np.uint8) * 255

    # Combine face with the white background using the circle mask
    face_circled = np.where(binary_mask[..., None] == 255, square_face, white_bg)

    return face_circled

def main(min_dim=256, margin_factor=1.2):
    """
    Main function to detect faces, expand bounding box, crop them into circles, and save.
    :param min_dim:        Minimum dimension (width/height) of final circle image
    :param margin_factor:  Scale factor for the bounding box (to avoid tight cropping)
    """
    image_path = "group_photo.jpg"
    cascade_path = "/root/haarcascade_frontalface_default.xml"  # Adjust to your path

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check path and file format.")
        return

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Could not load Haar cascade. Check .xml file path.")
        return

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Detected {len(faces)} face(s).")

    # Create output directory
    output_dir = "circle_profiles"
    os.makedirs(output_dir, exist_ok=True)

    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_circled = circle_crop_face(
            img, x, y, w, h, 
            min_dim=min_dim, 
            margin_factor=margin_factor
        )
        out_path = os.path.join(output_dir, f"face_{i}.png")
        cv2.imwrite(out_path, face_circled)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # You can tweak these defaults to your preference:
    #  - min_dim=256 ensures images are at least 256x256
    #  - margin_factor=1.2 means 20% bigger than the detected bounding box
    main(min_dim=256, margin_factor=1.2)