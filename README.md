# LSTM Text Generation (Generative AI)

This project implements a character-level LSTM model to generate text in the style of Shakespeare.

## Overview
- Dataset: Shakespeare’s Complete Works (Project Gutenberg)
- Model: Embedding + LSTM + Dense
- Framework: TensorFlow / Keras
- Approach: Character-level language modeling

## How it works
The model is trained to predict the next character given a sequence of previous characters. After training, it generates new text based on a seed input.

## Running the project
Open the notebook in Google Colab and run all cells. The dataset is downloaded automatically.

## Sample Output
The model generates Shakespeare-style text after training, demonstrating learned linguistic patterns.

## Notes
- Character-level models may produce repetitive text with limited epochs.
- Text diversity can be improved using temperature sampling.

## Dataset
Public domain text from Project Gutenberg:
https://www.gutenberg.org/files/100/100-0.txt
