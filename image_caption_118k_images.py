{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "_BSVz4F_VkWe"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Loading annotations to get image filenames...\n",
            "Found 118287 unique images in the dataset.\n",
            "Loading InceptionV3 model for feature extraction...\n",
            "Starting feature extraction. Features will be saved in 'features'...\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Extracting Features: 100%|██████████| 118287/118287 [3:21:17<00:00,  9.79it/s] "
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "✅ Feature extraction complete!\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n"
          ]
        }
      ],
      "source": [
        "# image_caption_118k_images.py (Updated)\n",
        "\n",
        "import os\n",
        "import json\n",
        "import string\n",
        "import numpy as np\n",
        "from tqdm import tqdm\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input\n",
        "from tensorflow.keras.models import Model\n",
        "\n",
        "# --- 1. SETUP PATHS ---\n",
        "BASE_DIR = \"coco\"\n",
        "ANNOTATIONS_FILE = os.path.join(BASE_DIR, \"annotations\", \"captions_train2017.json\")\n",
        "IMAGE_DIR = os.path.join(BASE_DIR, \"train2017\")\n",
        "FEATURES_DIR = \"features\" # Directory to save individual .npy feature files\n",
        "\n",
        "# Create the features directory if it doesn't exist\n",
        "if not os.path.exists(FEATURES_DIR):\n",
        "    os.makedirs(FEATURES_DIR)\n",
        "\n",
        "# --- 2. LOAD ANNOTATIONS TO GET IMAGE FILENAMES ---\n",
        "print(\"Loading annotations to get image filenames...\")\n",
        "with open(ANNOTATIONS_FILE, 'r') as f:\n",
        "    annotations = json.load(f)\n",
        "\n",
        "# Create a list of all unique image filenames\n",
        "image_files = sorted(list(set([img['file_name'] for img in annotations['images']])))\n",
        "print(f\"Found {len(image_files)} unique images in the dataset.\")\n",
        "\n",
        "# --- 3. LOAD THE PRE-TRAINED CNN MODEL ---\n",
        "print(\"Loading InceptionV3 model for feature extraction...\")\n",
        "base_model = InceptionV3(weights='imagenet')\n",
        "# Create a new model that outputs the feature vector from the penultimate layer\n",
        "cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)\n",
        "\n",
        "# --- 4. FEATURE EXTRACTION FUNCTION ---\n",
        "def extract_and_save_features(img_path, model):\n",
        "    \"\"\"Loads, preprocesses, and extracts features from an image.\"\"\"\n",
        "    try:\n",
        "        # Load and resize the image\n",
        "        img = image.load_img(img_path, target_size=(299, 299))\n",
        "        # Convert image to numpy array and add batch dimension\n",
        "        x = image.img_to_array(img)\n",
        "        x = np.expand_dims(x, axis=0)\n",
        "        # Pre-process for InceptionV3\n",
        "        x = preprocess_input(x)\n",
        "        # Extract features (shape will be (1, 2048))\n",
        "        features = model.predict(x, verbose=0)\n",
        "        return features\n",
        "    except Exception as e:\n",
        "        print(f\"Error processing {img_path}: {e}\")\n",
        "        return None\n",
        "\n",
        "# --- 5. MAIN PROCESSING LOOP ---\n",
        "print(f\"Starting feature extraction. Features will be saved in '{FEATURES_DIR}'...\")\n",
        "\n",
        "for filename in tqdm(image_files, desc=\"Extracting Features\"):\n",
        "    # Construct paths\n",
        "    image_path = os.path.join(IMAGE_DIR, filename)\n",
        "    feature_filename = filename.replace('.jpg', '.npy')\n",
        "    feature_path = os.path.join(FEATURES_DIR, feature_filename)\n",
        "\n",
        "    # Skip if feature file already exists\n",
        "    if os.path.exists(feature_path):\n",
        "        continue\n",
        "\n",
        "    # Extract and save features\n",
        "    if os.path.exists(image_path):\n",
        "        features = extract_and_save_features(image_path, cnn_model)\n",
        "        if features is not None:\n",
        "            np.save(feature_path, features)\n",
        "    else:\n",
        "        print(f\"Warning: Image file not found: {image_path}\")\n",
        "\n",
        "print(\"\\n✅ Feature extraction complete!\")"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": ".venv",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.1"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
