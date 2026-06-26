# Sentiment Analysis API with Conv1D Deep Learning

A production-oriented Natural Language Processing (NLP) project that performs sentiment analysis on text using a **1D Convolutional Neural Network (Conv1D)** built with TensorFlow/Keras.

The project includes:

-  End-to-end text preprocessing
-  Tokenization and sequence padding
-  Conv1D-based deep neural network for sentiment classification
-  Model serialization and loading for inference
-  Dockerized deployment
-  Clean project structure for future experimentation and production use

This repository serves as a foundation for experimenting with increasingly advanced sequence models such as **LSTMs** and **Transformer-based architectures**.

---

## Features

- Deep learning sentiment classification
- Conv1D neural network architecture
- TensorFlow/Keras implementation
- Saved tokenizer and trained model
- Modular training and inference scripts
- Docker support for reproducible deployment
- Easy to extend with new architectures

---

## Project Structure

```text
Sentiment_app/
│
├── app/
│   ├── data/
│   │   └── IMDB_Dataset.csv
│   │
│   ├── model/
│   │   ├── cnn_sentiment_v3.keras
│   │   └── tokenizer.json
│   │
│   ├── data.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── main.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## Model Architecture

Current implementation uses a **Conv1D (1-Dimensional Convolutional Neural Network)**.

Typical pipeline:

```
Input Text
      │
      ▼
Tokenizer
      │
      ▼
Integer Sequences
      │
      ▼
Padding
      │
      ▼
Embedding Layer
      │
      ▼
Conv1D
      │
      ▼
Global Max Pooling
      │
      ▼
Dense Layers
      │
      ▼
Sigmoid Output
      │
      ▼
Positive / Negative Sentiment
```

Conv1D models efficiently capture local word patterns and are computationally lightweight compared to recurrent models.

---

## Dataset

The project is trained on the **IMDb Movie Reviews Dataset**, a benchmark dataset for binary sentiment classification containing positive and negative movie reviews.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EddiesEdit/sentiment-analysis-api.git
cd  sentiment-analysis-api
```

Create a virtual environment:

### Linux / WSL

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🎬 Demo

Watch the complete walkthrough of the Sentiment Analysis API, including model inference, FastAPI endpoints, Swagger UI testing, and Docker deployment.

[![Watch Demo](assets/demo-thumbnail.png)](https://youtu.be/AYDxyyB3vA8)

> Click the image above to watch the full demonstration on YouTube.
---

---

## Training

Train the Conv1D model:

```bash
python app/train.py
```

The trained model and tokenizer will be saved for inference.

---

## Inference

Run predictions on new text:

```bash
python app/inference.py
```

Or start the application:

```bash
python app/main.py
```

---

## Docker

Build the Docker image:

```bash
docker build -t sentiment-app .
```

Run the container:

```bash
docker run -p 8000:8000 sentiment-app
```

Docker ensures a consistent environment across development and deployment.

---

## Future Improvements

This project is actively evolving. Planned enhancements include:

- [ ] Bidirectional LSTM implementation
- [ ] GRU-based sequence models
- [ ] Transformer-based architectures
- [ ] BERT fine-tuning
- [ ] Attention mechanisms
- [ ] Hyperparameter optimization
- [ ] Experiment tracking
- [ ] CI/CD integration
- [ ] Cloud deployment
- [ ] Model monitoring

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Docker

---

## Project Goals

The objectives of this project are to:

- Build a robust sentiment analysis system using deep learning.
- Demonstrate end-to-end NLP workflows.
- Provide a reproducible and deployable machine learning pipeline.
- Establish a baseline architecture for future experiments with LSTM and Transformer models.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.