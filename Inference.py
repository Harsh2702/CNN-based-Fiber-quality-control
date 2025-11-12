import torch
import cv2
import numpy as np

# -------------------------------
# 1. Load the trained model
# -------------------------------
model = torch.load("model.pth")
model.eval()  # set to evaluation mode

# -------------------------------
# 2. Load and preprocess image
# -------------------------------
img_path = "image.jpg"  # path to your test image
image = cv2.imread(img_path)

if image is None:
    raise ValueError(f"Image not found: {img_path}")

# Convert BGR (OpenCV) → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

# Transpose to (C, H, W)
image = np.transpose(image, (2, 0, 1))

# Convert to float32 and normalize 0–1
image = image.astype(np.float32) / 255.0

# Convert to tensor and add batch dimension
input_tensor = torch.from_numpy(image).unsqueeze(0)  # shape: (1, 3, 512, 512)

# -------------------------------
# 3. Make prediction
# -------------------------------
with torch.no_grad():
    output = model(input_tensor)

# If model outputs class logits:
pred = torch.argmax(output, dim=1).item()

print("Predicted class index:", pred)

# Optional: map to class name
# class_names = ["Good", "Bad"]
# print("Predicted label:", class_names[pred])
