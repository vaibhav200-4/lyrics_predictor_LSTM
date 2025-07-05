import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = load_model("lyrics_saphire.keras")

# Word prediction function
def wordpred(model, tokenizer, seed_text, next_words, max_seq_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0))
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Streamlit app UI
st.title("Next Word Prediction with LSTM RNN")

input_text = st.text_input("Enter the sequence of words", "pushing on your body")
next_words_count = st.number_input("How many words to predict?", min_value=1, max_value=100, value=1, step=1)

if st.button("Predict Next Word(s)"):
    max_seq_len = model.input_shape[1] + 1
    result_text = wordpred(model, tokenizer, input_text, next_words_count, max_seq_len)
    st.write(f"**Generated Text:** {result_text}")
import streamlit as st

