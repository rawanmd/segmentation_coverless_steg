# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/rawanmd/segmentation_coverless_steg/blob/adaptive_window/Segmentation_Coverless_Steg.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="alEU1axrq2-w"
from textwrap import indent
from google.colab import drive

import os
import glob
import json

# %% [markdown] id="WjB_xCHxJL4C"
# ## Connect to Drive

# %% colab={"base_uri": "https://localhost:8080/"} id="D6pNBetB1ebx" outputId="debccb37-1bee-4f7c-bc52-aad11cbfd492"
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Dataset root
dataset_dir = "/content/drive/MyDrive/coco_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Move into dataset folder
# %cd $dataset_dir

# Create desired folders
os.makedirs("annotations", exist_ok=True)
os.makedirs("train", exist_ok=True)
os.makedirs("val", exist_ok=True)

# ==========================
# 1. ANNOTATIONS
# ==========================
# # !wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# # !unzip -q annotations_trainval2017.zip
# # !rm annotations_trainval2017.zip

# # ==========================
# # 2. VALIDATION IMAGES
# # ==========================
# # !wget -c http://images.cocodataset.org/zips/val2017.zip

# # !unzip -q val2017.zip
# # !rm val2017.zip

# # Move images into val/
# # !mv val2017/* val/
# # !rmdir val2017

# ==========================
# 3. TRAIN IMAGES
# ==========================
# # !wget -c http://images.cocodataset.org/zips/train2017.zip

# # !unzip -q train2017.zip
# # !rm train2017.zip

# # Move images into train/
# # !mv train2017/* train/
# # !rmdir train2017

# %% [markdown] id="0ellvUphd1IU"
# ## Import SAM

# %% colab={"base_uri": "https://localhost:8080/"} id="5b4D_cNmfszT" outputId="04c1d80c-49ad-4c2d-c299-cf9d3357d169"
# !pip install git+https://github.com/facebookresearch/segment-anything.git
# !pip install opencv-python pycocotools matplotlib
# !pip install torch torchvision

import torch
import cv2
import matplotlib.pyplot as plt

# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# %% id="uqmiR5ByeG5D"
MODEL_TYPE = "vit_b"
CHECKPOINT = "sam_vit_b_01ec64.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)


# %% [markdown] id="pjruehLAzgwf"
# ## Functions

# %% id="p9kSYrulYYLT"
def extract_masks(image_path):

  import math

  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  masks = mask_generator.generate(image)

  binary_seq = ""

  for i in range(len(masks) - 1):
    bbox1 = masks[i]['bbox']
    x1, y1 = bbox1[:2]

    bbox2 = masks[i+1]['bbox']
    x2, y2 = bbox2[:2]

    d_squared = (x2 - x1)**2 + (y2 - y1)**2
    d = math.sqrt(d_squared)

    if(d <= 300):
      binary_seq += '0'
    else:
      binary_seq += '1'

  return binary_seq, len(binary_seq)

def get_image_paths(dir):
  image_paths = sorted(glob.glob(os.path.join(dir, '*.jpg')))
  print(f"Found {len(image_paths)} images.")
  return image_paths

def build_index(image_paths):
  index = {}

  for image_path in image_paths:
    # call segmentation model
    binary_seq, length = extract_masks(image_path)

    if binary_seq is None:
      continue

    if binary_seq not in index:
      index[binary_seq] = []

    index[binary_seq].append({"image path": image_path,
                              "length": length})

    with open(str(dataset_dir)+"/inverted_index.json", "w") as f:
      json.dump(index, f, indent=4)

  return index


# %% id="BbA3BWjOp05k"
def match_msg_to_images(binary_msg, window_size, index):
  while window_size > 0:

      image_paths = []
      success = True

      print(f"\nTrying window size: {window_size}")

      for i in range(0, len(binary_msg), window_size):

          chunk = binary_msg[i:i + window_size]
          num_zeros = 0

          # add trailing zeros to last chunk
          if len(chunk) < window_size:
              num_zeros = window_size - len(chunk)
              chunk = chunk.ljust(window_size, '0')

          print(f"chunk: {chunk}")

          key = ''

          for idx in index:
              if idx.startswith(chunk):
                  key = idx
                  break

          if key == '':
              print(f"No match for {chunk}")
              success = False
              break

          image_paths.append(index[key][0]['image path'])
          print(f"key: {key}")

      if success:
          print(f"image paths: {image_paths}")
          print(f"Final window size: {window_size}")
          return image_paths, window_size, num_zeros

      window_size -= 1

  return [], -1, -1


# %% [markdown] id="e8JvDxal1iZc"
# ## Pipeline

# %% id="DpiK-TGz1tsQ"
PATH = "/content/drive/MyDrive/coco_dataset/train2017"

secret_msg = "Hello World"
ascii_secret_msg = [ord(char) for char in secret_msg]

# add text enc here

binary_msg = [f"{val:08b}" for val in ascii_secret_msg]
binary_msg = ''.join(binary_msg)

# build index if not already there
# index = build_index(PATH)

# load index if it exists
index = None
with open("/content/drive/MyDrive/coco_dataset/inverted_index.json", "r") as f:
  index = json.load(f)

# %% colab={"base_uri": "https://localhost:8080/"} id="v27igmcqyAwP" outputId="6729349b-9f34-4286-8815-91d349fe1dd7"
window_size = len(max(index.keys(), key=len))
print(f"Max window size in index: {window_size}")

# all_keys, w_size, num_zeros = match_msg_to_images(binary_msg, window_size, index)
all_keys, w_size, num_zeros = match_msg_to_images('1101111', 4, index)  # just an example for tracing

# %% id="8jWsyk0rpmns"
