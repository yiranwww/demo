import numpy as np


def calculate_contrast(img) -> int:
      min_val = np.min(img)
      max_val = np.max(img)
      contrast = (max_val-min_val)
      return contrast


img = np.array([[0, 50], [200, 255]])
contrast = calculate_contrast(img)
print(contrast) 