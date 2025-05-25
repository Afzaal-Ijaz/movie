import streamlit as st
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model and tokenizer
#
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load trained Keras model
model = load_model('gru_model.h5')

# Streamlit app setup
st.title('Sentiment Analysis App')
st.write('Enter a sentence to analyze its sentiment.')
# User input
user_input = st.text_area('Input your text here:')
# Analyze button
if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess the input
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=100)  # Adjust maxlen as needed
        # Make prediction
        prediction = model.predict(input_padded)
        sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'

   # Display the result
        st.success(f'The sentiment is: {sentiment}')
    else:
        st.error('Please enter some text to analyze.')