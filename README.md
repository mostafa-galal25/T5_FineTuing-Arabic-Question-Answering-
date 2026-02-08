# T5 Fine-Tuning for Arabic Question Answering

This repository contains a Jupyter Notebook for fine-tuning a **T5 (Text-to-Text Transfer Transformer)** model on an **Arabic Question Answering** task using the **TyDi QA Arabic dataset**.  
The project is implemented using **PyTorch**, **Hugging Face Transformers**, and **PyTorch Lightning**.

---

## 📌 Project Overview

The goal of this project is to fine-tune a pre-trained T5 model to generate accurate answers for Arabic questions based on given contexts.  
The notebook demonstrates:
- Data preprocessing
- Custom dataset creation
- Model fine-tuning
- Training and validation using PyTorch Lightning
- Model checkpointing and logging

---

## 📂 Dataset

- **Dataset**: TyDi QA (Arabic)
- **Format**: JSON  
- **Example file used**:tydiqa-goldp-v1.1-train-ar.json

The dataset contains:
- Context passages
- Questions
- Ground-truth answers

---

## 🧠 Model

- **Model**: T5 (Text-to-Text Transfer Transformer)
- **Tokenizer**: `T5Tokenizer`
- **Architecture**: `T5ForConditionalGeneration`

---

## 🛠️ Technologies & Libraries

- Python
- PyTorch
- PyTorch Lightning
- Hugging Face Transformers
- Pandas
- Scikit-learn
- TensorBoard

---

## ⚙️ Training Pipeline

1. Load and preprocess the Arabic TyDi QA dataset
2. Split data into training and validation sets
3. Create a custom `Dataset` and `DataLoader`
4. Fine-tune the T5 model using PyTorch Lightning
5. Monitor training with TensorBoard
6. Save best model checkpoints automatically

---

Open the notebook:  T5_FineTuing.ipynb

Install required dependencies: pip install torch transformers pytorch-lightning pandas scikit-learn
📈 Outputs

Fine-tuned T5 model

Saved checkpoints

Training & validation loss logs

TensorBoard logs for visualization

🔮 Future Improvements

Add evaluation metrics (Exact Match, F1-score)

Support more Arabic QA datasets

Export the trained model for inference

Build an API or chatbot interface

👤 Author

Mostafa Galal
Data Scientist | NLP Enthusiast
