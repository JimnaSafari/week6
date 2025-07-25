import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import random

# --- Model and Class Names ---
MODEL_PATH = "garbage_model.tflite"
CLASS_NAMES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMG_SIZE = (224, 224)  # Adjust if your model uses a different size

# --- Quiz questions and fun facts ---
QUIZ_QUESTIONS = [
    {
        "question": "Which bin should a plastic bottle go in?",
        "options": ["Recycling", "Compost", "Trash"],
        "answer": "Recycling"
    },
    {
        "question": "How long does it take for glass to decompose?",
        "options": ["1 year", "100 years", "1 million years"],
        "answer": "1 million years"
    },
    {
        "question": "Can pizza boxes be recycled if they are greasy?",
        "options": ["Yes", "No"],
        "answer": "No"
    },
    {
        "question": "Which material can be recycled endlessly without loss in quality?",
        "options": ["Plastic", "Glass", "Cardboard"],
        "answer": "Glass"
    },
    {
        "question": "What should you do before recycling a metal can?",
        "options": ["Rinse it", "Crush it", "Nothing"],
        "answer": "Rinse it"
    },
]

FUN_FACTS = [
    "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
    "Glass can be recycled endlessly without loss in quality.",
    "Plastic bags can take up to 1,000 years to decompose in landfills.",
    "Cardboard is one of the easiest materials to recycle.",
    "Metal recycling reduces the need for mining new ore, saving resources.",
    "Paper can only be recycled 5-7 times before the fibers become too short."
]

# --- Load TFLite model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# --- Prediction function ---
def predict_image(image, interpreter):
    img = Image.open(image).convert("RGB").resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return CLASS_NAMES[np.argmax(output_data)]

# --- Streamlit App ---
st.set_page_config(page_title="Recyclable Item Classifier + Quiz", page_icon="♻️")
st.title("♻️ Recyclable Item Classifier + Quiz")

# Session state for score
if "score" not in st.session_state:
    st.session_state.score = 0
if "questions_answered" not in st.session_state:
    st.session_state.questions_answered = 0
if "last_quiz" not in st.session_state:
    st.session_state.last_quiz = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "show_fact" not in st.session_state:
    st.session_state.show_fact = False

interpreter = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload an image of a recyclable item", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    prediction = predict_image(uploaded_file, interpreter)
    st.success(f"Prediction: {prediction}")

    # Show a quiz question
    if st.session_state.last_quiz is None or st.session_state.show_fact:
        quiz = random.choice(QUIZ_QUESTIONS)
        st.session_state.last_quiz = quiz
        st.session_state.last_answer = None
        st.session_state.show_fact = False
    else:
        quiz = st.session_state.last_quiz

    st.write("### Quiz Time!")
    user_answer = st.radio(quiz["question"], quiz["options"], key=st.session_state.questions_answered)
    if st.button("Submit Answer", key=st.session_state.questions_answered):
        st.session_state.questions_answered += 1
        st.session_state.last_answer = user_answer
        if user_answer == quiz["answer"]:
            st.session_state.score += 1
            st.success("Correct! 🎉")
        else:
            st.error(f"Incorrect. The correct answer is: {quiz['answer']}")
        st.session_state.show_fact = True
        st.info("Fun Fact: " + random.choice(FUN_FACTS))

    # Show score
    st.write(f"**Score:** {st.session_state.score} / {st.session_state.questions_answered}")

st.write("---")
st.write("Upload more images to keep playing and learning!") 