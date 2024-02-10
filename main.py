import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Initialize a set to store the stop words
stop_words = set()

# List of stop words files
stop_words_files = [
    'StopWords_Auditor.txt',
    'StopWords_Currencies.txt',
    'StopWords_DatesandNumbers.txt',
    'StopWords_Generic.txt',
    'StopWords_GenericLong.txt',
    'StopWords_Geographic.txt',
    'StopWords_Names.txt'
]

# Load stop words from each file
print("Loading stop words...")
for file_name in stop_words_files:
    with open(os.path.join('StopWords', file_name), 'r') as f:
        words = f.read().splitlines()
        stop_words.update(words)

print("Stop words loaded.")

# Load positive and negative words
print("Loading positive and negative words...")
with open('positive-words.txt', 'r') as f:
    positive_words = set(word.strip() for word in f if word.strip() not in stop_words)

with open('negative-words.txt', 'r') as f:
    negative_words = set(word.strip() for word in f if word.strip() not in stop_words)

print("Positive and negative words loaded.")

# Load the data
print("Loading data from 'Input.xlsx'...")
data = pd.read_excel('Input.xlsx')
print("Data loaded.")

# Load the Output file if it exists
output_file = 'Output.csv'
if os.path.exists(output_file):
    output = pd.read_csv(output_file)
else:
    output = pd.DataFrame(
        columns=['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
                 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
                 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])


# Function to count syllables in a word
def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count


# Loop over each URL
for index, row in data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Skip this URL if it has already been processed
    if url in output['URL'].values:
        print(f"Skipping URL: {url}")
        continue

    print(f"Processing URL: {url}")

    # Send a request to the website
    r = requests.get(url)
    r.content

    # Parse the HTML content
    soup = BeautifulSoup(r.content, 'html.parser')

    # Extract title and text
    title = soup.find('title').string
    text = soup.get_text()

    # Remove stop words from the text
    cleaned_text = ' '.join(word for word in text.split() if word not in stop_words)

    # Perform text analysis
    blob = TextBlob(cleaned_text)
    positive_score = sum(1 for word in cleaned_text.split() if word in positive_words)
    negative_score = sum(1 for word in cleaned_text.split() if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / ((len(cleaned_text.split())) + 0.000001)

    sentences = sent_tokenize(cleaned_text)
    words = word_tokenize(cleaned_text)
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
    complex_words = [w for w in words if count_syllables(w) > 2]
    percentage_of_complex_words = len(complex_words) / len(words)
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    avg_number_of_words_per_sentence = len(words) / len(sentences)
    complex_word_count = len(complex_words)
    word_count = len(words)
    syllable_per_word = sum(count_syllables(w) for w in words) / len(words)
    personal_pronouns = len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', cleaned_text, re.I))
    avg_word_length = sum(len(w) for w in words) / len(words)

    # Create a new DataFrame with the results
    new_row = pd.DataFrame({
        'URL_ID': [url_id],
        'URL': [url],
        'POSITIVE SCORE': [positive_score],
        'NEGATIVE SCORE': [negative_score],
        'POLARITY SCORE': [polarity_score],
        'SUBJECTIVITY SCORE': [subjectivity_score],
        'AVG SENTENCE LENGTH': [avg_sentence_length],
        'PERCENTAGE OF COMPLEX WORDS': [percentage_of_complex_words],
        'FOG INDEX': [fog_index],
        'AVG NUMBER OF WORDS PER SENTENCE': [avg_number_of_words_per_sentence],
        'COMPLEX WORD COUNT': [complex_word_count],
        'WORD COUNT': [word_count],
        'SYLLABLE PER WORD': [syllable_per_word],
        'PERSONAL PRONOUNS': [personal_pronouns],
        'AVG WORD LENGTH': [avg_word_length]
    })

    # Append the new row to the progress DataFrame
    output = pd.concat([output, new_row], ignore_index=True)
    # Save the progress DataFrame to a file
    output.to_csv(output_file, index=False)

print("All URLs processed.")
