import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import layers

# Define a small text corpus for demonstration
corpus = (
    "I went to the park the other day with my parents and I enjoyed the fresh air. "
    "I hurt my foot while playing football with my friends in the park but I kept playing. "
    "Basketball is a sport that requires agility, coordination, and teamwork to succeed. "
    "The game brings excitement and passion to both players and fans alike."
)

# Prepare a tokenizer and fit it on the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
total_words = len(tokenizer.word_index) + 1  # Vocabulary size

# Create input sequences from the corpus using a sliding window of length 5 + 1 target word
seq_length = 5
tokens = tokenizer.texts_to_sequences([corpus])[0]
input_sequences = []
for i in range(seq_length, len(tokens)):
    n_gram_seq = tokens[i - seq_length:i + 1]
    input_sequences.append(n_gram_seq)
input_sequences = np.array(input_sequences)

# Split sequences into inputs (X) and target (y)
X_seq = input_sequences[:, :-1]
y_seq = to_categorical(input_sequences[:, -1], num_classes=total_words)

# Define architectures to test: RNN, LSTM, GRU
architectures = ["RNN", "LSTM", "GRU"]
results = {}

for arch in architectures:
    # Build the text generation model
    text_model = Sequential()
    text_model.add(layers.Embedding(total_words, 10, input_length=seq_length))
    if arch == "RNN":
        text_model.add(layers.SimpleRNN(50))
    elif arch == "LSTM":
        text_model.add(layers.LSTM(50))
    elif arch == "GRU":
        text_model.add(layers.GRU(50))
    else:
        raise ValueError("Unknown model type")
    text_model.add(layers.Dense(total_words, activation="softmax"))
    text_model.compile(loss="categorical_crossentropy", optimizer="adam")
    
    # Train the text generation model for 200 epochs (training is silent with verbose=0)
    text_model.fit(X_seq, y_seq, epochs=200, verbose=0)
    
    # Prepare seed prompts to generate text
    seeds = {
        "prompt1": "I went to the park the other day with my parents and I",  # generate 5 extra words
        "prompt2": ("I hurt my foot the other while playing football with my friends in the park, "
                    "but I still wanted to keep playing as I was having a lot of fun. Then I came home and I ate a lot of carbonara that I had prepared yesterday so now my belly hurts and so does my"),
        "prompt3": "Basketball is"  # generate 50 extra words
    }
    results[arch] = {}
    
    # For each prompt, determine how many extra words to generate and produce text
    for key, seed in seeds.items():
        if key == "prompt1":
            extra = 5
        elif key == "prompt2":
            extra = 1
        else:
            extra = 50

        seed_text = seed
        for _ in range(extra):
            # Convert the current text to a sequence of integers
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            # Pad the sequence to ensure it has the required length
            token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
            # Predict the next word
            predicted = np.argmax(text_model.predict(token_list, verbose=0), axis=-1)[0]
            # Find the word corresponding to the predicted index and append it
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    seed_text += " " + word
                    break
        results[arch][key] = seed_text
        print(f"\n[{arch}] {key}:\n{seed_text}\n")

# Finally, print all generated text results for each architecture
print("\nResultados de generación de texto:")
for arch, prompts in results.items():
    print(f"\nModelo {arch}:")
    for key, text in prompts.items():
        print(f"\n[{key}]:\n{text}")