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
# <a href="https://colab.research.google.com/github/rawanmd/segmentation_coverless_steg/blob/main/Segmentation_Coverless_Steg.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="alEU1axrq2-w"
from textwrap import indent
from google.colab import drive

import os
import glob
import json

# %% colab={"base_uri": "https://localhost:8080/"} id="D6pNBetB1ebx" outputId="08a1f8ba-f499-430c-e4fd-098ac088e932"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="5b4D_cNmfszT" outputId="e1d4abb6-6841-4fe6-869a-3ecfdb8fe04e"
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


# %% [markdown] id="pjruehLAzgwf"
# ## Create the Inverted Index

# %% id="p9kSYrulYYLT"
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


# %% [markdown] id="e8JvDxal1iZc"
# ## Pipeline

# %% [markdown] id="AHZ4MDXE0fyA"
# ### Text Encryption

# %% [markdown] id="kBA-n0Hf0nVi"
# ### Divide and match the message to images

# %% id="DpiK-TGz1tsQ"
PATH = "/content/drive/MyDrive/coco_dataset/train2017"

secret_msg = "Hello World"
ascii_secret_msg = [ord(char) for char in secret_msg]

# add text enc here

binary_msg = [f"{val:08b}" for val in ascii_secret_msg]

# build index if not already there
# index = build_index(PATH)

# load index if it exists
index = None
with open("/content/drive/MyDrive/coco_dataset/inverted_index.json", "r") as f:
  index = json.load(f)



# %% colab={"base_uri": "https://localhost:8080/"} id="Zk6yeACrrOBr" outputId="7079c13c-27ac-4b07-ea14-fd606ef58c88"
window_size = max(len(key) for key in index.keys())
print("Largest length:", window_size)

for chunk in binary_msg:


# %% colab={"base_uri": "https://localhost:8080/"} id="YDlaXHFdsUJt" outputId="10a80ce6-0e78-4aba-81c3-72ab7db68268"
window_size = 10

prefix_map = {}

for code in index.keys():
    prefix = code[:window_size]

    if prefix not in prefix_map:
        prefix_map[prefix] = []

    prefix_map[prefix].append(code)

prefix_map["0110011010"]

# %% colab={"base_uri": "https://localhost:8080/"} id="MHmBE3rGeKUi" outputId="5a3cc4fb-a6d4-4dcb-c817-5ac53dd07487"
all_codes = list(index.keys())

print(index.keys())

prefix_map = {}

for window_size in range(10):
  for code in all_codes:
      # window_size = index[code][0]['length']
      prefix = code[:window_size]
      if prefix not in prefix_map:
          prefix_map[prefix] = []

      prefix_map[prefix].append(code)

prefix_map

# %% id="CzFAJoHQeT4z"
