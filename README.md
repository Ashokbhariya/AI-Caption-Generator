🌟 AI Caption Generator

Generate intelligent, context-aware captions for images using deep learning (CNN + LSTM / Transformer models).

🚀 Overview

The AI Caption Generator is a deep learning–based project that automatically generates meaningful captions for images.
It combines Convolutional Neural Networks (CNNs) for image feature extraction and sequence models (LSTM / Transformer) for generating natural language captions.

This project demonstrates practical AI skills including computer vision, NLP, and model deployment.

✨ Features

📸 Extracts image features using a pretrained CNN model (ResNet / Inception / VGG).

🧠 Generates captions using LSTM or Transformer-based decoder.

📊 Supports datasets like Flickr8k, Flickr30k, COCO.

🧹 Includes preprocessing: tokenization, padding, vocabulary building.

🚀 Train, evaluate, and generate captions easily.

🌐 Optional API support (FastAPI) to generate captions via endpoint.

🏗️ Project Architecture
Image → CNN Encoder → Feature Vector → LSTM/Transformer Decoder → Caption

📁 Folder Structure
├── data/
│   ├── images/
│   ├── captions.txt
├── models/
│   ├── encoder.h5
│   ├── decoder.h5
├── notebooks/
│   ├── training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── utils.py
├── api/
│   ├── main.py (optional FastAPI app)
├── README.md

🛠️ Tech Stack

Languages: Python
Libraries: TensorFlow/Keras, NumPy, Pandas, Matplotlib, OpenCV
Tools: Jupyter Notebook, VS Code
Dataset: Flickr8k / COCO

⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/ai-caption-generator.git
cd ai-caption-generator

2. Install Dependencies
pip install -r requirements.txt

📦 Dataset Setup

Download any dataset:

Flickr8k

Flickr30k

MS COCO

Place:

/data/images  
/data/captions.txt  


Update paths in config.py as needed.

🧠 Model Training

Run preprocessing:

python src/preprocess.py


Train the model:

python src/train.py

🔮 Generate Captions
python src/inference.py --image path/to/image.jpg

🌐 (Optional) API Usage

Start FastAPI server:

uvicorn api.main:app --reload


Send request:

POST /generate-caption
{
  "image_url": "..."
}

🎯 Future Improvements

Integrate Transformers (ViT + GPT-style decoder)

Add multilingual captioning

Build a web UI

Deploy model on Streamlit / FastAPI

🤝 Contributing

Contributions are welcome!
Feel free to open issues or create pull requests.

📜 License

This project is licensed under the MIT License.
