# Build a streamlit interface for our classifier.
import streamlit as st
from streamlit_backend import *
st.title("An Emotion Classification Tool")
st.header("Let the model guess the emotion in a text!")
reminder = "Please type your text below:"
user_input = st.text_input(reminder, "Hello, how are you doing?")

EMOTION_DICT = {0: ("anger", '😠'), 
                1: ("disapproval", '❌'),
                2: ("disgust", '🤢'),
                3: ("fear", '😨'),
                4: ("happy", '😄'),
                5: ("sadness", '😫'),
                6: ("surprise", '🤯'),
                7: ("neutral", '😑')}

# Feed to the tokenizer
if len(user_input) > 0:
    model, text2Vec_Layer = init_model()
    predicted_label, predicted_prob = get_prediction(model, text2Vec_Layer, user_input[0])
    emotion, emoji = EMOTION_DICT[predicted_label]
    st.text("With prediction probability: " + "{:.3f}".format(predicted_prob))
    st.text("We think the emotion is " + emotion + " " + emoji + ".")
    st.caption("ALERT: The model is based on a single transformer block. It took only 3 minutes to train 20 epochs. It's a tiny model that can deploy on this platform. The accuracy is much lower than the fine-tuned BERT model. Please also check out the fine-tuned BERT model on HuggingFace Hub.")