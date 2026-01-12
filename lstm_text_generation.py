
"""
Original file is located at
    https://colab.research.google.com/drive/1fft3iviOSGxCI3GXwzcQsYfuka5q1GJ-
"""

import tensorflow as tf
import numpy as np
import requests
import re
import string

url = "https://www.gutenberg.org/files/100/100-0.txt" #My dataset
text = requests.get(url).text.lower()     #it's covert all the data into lower case


start_marker = "*** start of the project gutenberg ebook the complete works of william shakespeare ***"   
end_marker = "*** end of the project gutenberg ebook ***"

start_idex = text.find(start_marker)
end_idex = text.find(end_marker)


if start_idex != -1 and end_idex != -1 and start_idex < end_idex:
   
    text = text[start_idex + len(start_marker):end_idex]
else:
    #it print warning if marker are not found 
    print("Warning: Start or end markers not found or in incorrect order.")

# text cleaning 
text = re.sub(f"[{re.escape(string.punctuation)}][^a-z]", "", text) # Changed regex to keep alphabets


chars = sorted(list(set(text)))
char_to_idex = {c: i for i, c in enumerate(chars)}
idex_to_char = {i: c for i, c in enumerate(chars)}

encoded = np.array([char_to_idex[c] for c in text])


# Create sequences

SEQ_LEN = 40
X, y = [], []

for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i + SEQ_LEN])
    y.append(encoded[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)


# Build LSTM model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)


# Train model

model.fit(X, y, batch_size=128, epochs=5)


# Text generation

def generate_text(seed, length=300):
    seed = seed.lower()
    result = seed

    for _ in range(length):
        input_seq = np.array([[char_to_idex[c] for c in seed]])
        preds = model.predict(input_seq, verbose=0)
        next_char = idex_to_char[np.argmax(preds)]
        result += next_char
        seed = seed[1:] + next_char

    return result

print("\n***Generated Text***\n")
print(generate_text(text[:40]))