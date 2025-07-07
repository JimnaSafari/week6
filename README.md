# Recyclable-Item-Classifier-plpweek6

# ♻️ Edge AI Garbage Classification with TensorFlow Lite

This project implements an **Edge AI image classification model** to detect types of garbage (e.g., paper, plastic, metal, glass, cardboard, trash) using a lightweight Convolutional Neural Network (CNN). The model is trained using the **Garbage Classification Dataset** from Kaggle and converted to **TensorFlow Lite (TFLite)** for deployment on edge devices like Raspberry Pi or smartphones.

---

## 📌 Key Features

- 🧠 **CNN Model** trained on real-world garbage image data
- 💾 **Converted to TFLite** format for edge deployment
- 📊 Includes performance evaluation: accuracy, confusion matrix, classification report
- 🌐 **Streamlit UI**: Upload images and get predictions via web interface
- 📸 Optional synthetic image generation for data augmentation

---

## 🚀 Demo

Try the model in the Streamlit UI:

streamlit run app.py

less
Copy
Edit

Upload an image (e.g., plastic bottle) and get an instant prediction.

---

## 📁 Dataset

- Source: [Garbage Classification Dataset with Split](https://www.kaggle.com/datasets/naveenv070908/garbage-classification-with-split-train-and-test)
- Classes:
  - `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

To download:
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/naveenv070908/garbage-classification-with-split-train-and-test")
🛠 Tech Stack
Python 3.11

TensorFlow / TensorFlow Lite

Streamlit (for UI)

Google Colab (for training)

Seaborn, scikit-learn (for evaluation)

📦 How to Use
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/edge-ai-garbage-classifier.git
cd edge-ai-garbage-classifier
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Train and Convert Model (Google Colab Recommended)
Run the edge_model_training.ipynb notebook

This will train the model and save garbage_model.tflite

4. Launch the Streamlit App you will see a url like this 
bashhttps://twelve-buttons-feel.loca.lt/
Copy
Edit
streamlit run app.py
5. Upload Test Images
Use sample test images from the dataset or your own recyclable images.

📈 Results
Metric	Value
Training Accuracy	~90%
Validation Accuracy	~85%
Classes Supported	6
TFLite Model Size	~1.3MB

Includes confusion matrix and class-wise performance.

📂 Project Structure
pgsql
Copy
Edit
edge-ai-garbage-classifier/
│
├── app.py                       # Streamlit web app
├── garbage_model.tflite         # Trained TFLite model
├── edge_model_training.ipynb    # Training + conversion code
├── synthetic/                   # Augmented images (optional)
├── data/                        # Sample test data (optional)
├── README.md                    # You are here
└── requirements.txt             # Dependencies
🧠 Future Work
Optimize model size for microcontrollers (e.g., TinyML)

Real-time webcam support in Streamlit

Add multilingual prediction labels

📜 License
This project is licensed under the MIT License. Feel free to fork and improve!

🌍 Related Projects
TensorFlow Lite Microcontrollers

Streamlit Image Classifier Example

🙌 Acknowledgements
Kaggle Dataset Contributors

TensorFlow tutorials for image classification

yaml
Copy
Edit

---

### ✅ Want this as a `.md` file download?
I can generate it and give you a downloadable link.

Let me know if you'd also like:
- `requirements.txt`
- Auto-uploaded sample test images
- GitHub push instructions

Ready when you are!
