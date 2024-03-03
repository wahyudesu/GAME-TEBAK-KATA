import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load pre-trained BERT model and tokenizer for Bahasa Indonesia
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
model.eval()

# Dataset of Indonesian words (dummy data for demonstration)
indonesian_words = ["makan", "minum", "tidur", "berjalan", "belajar", "berbicara", "bermain", "membaca", "menulis", "memasak"]

# Function to select a random word from the dataset
def select_random_word():
    return random.choice(indonesian_words)

# Function to get BERT embeddings for a word
def get_bert_embedding(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    word_embedding = torch.mean(embeddings, dim=1).numpy()
    return word_embedding

# Function to calculate cosine similarity between two word embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# Main game loop
play_game = True
while play_game:
    # Select a random word from the dataset
    target_word = select_random_word()
    #print("Tebak kata: ", target_word)

    # Get BERT embedding for the target word
    target_embedding = get_bert_embedding(target_word)

    # Generate clue: You can customize how you want to generate the clue
    clue = target_word[:2]  # For example, the clue is the first two letters of the target word
    print("Petunjuk:", clue)

    # Loop for guessing the word
    while True:
        # Allow the user to guess the word
        guessed_word = input("Masukkan tebakan Anda (atau 'menyerah' untuk keluar): ")

        if guessed_word.lower() == "menyerah":
            play_game = False
            break

        # Get BERT embedding for the guessed word
        guessed_embedding = get_bert_embedding(guessed_word)

        # Calculate cosine similarity between the target word and guessed word
        similarity_score = calculate_cosine_similarity(target_embedding, guessed_embedding)

        # Display the similarity score
        print("Skor kesamaan:", similarity_score)

        # Check if the guessed word is correct
        if guessed_word == target_word:
            print("Selamat! Anda berhasil menebak kata.")
            break
        else:
            print("Tebakan Anda salah. Coba lagi.")

    # Ask if the user wants to play again
    if play_game:
        play_again = input("Ingin bermain lagi? (y/n): ")
        if play_again.lower() != 'y':
            break
