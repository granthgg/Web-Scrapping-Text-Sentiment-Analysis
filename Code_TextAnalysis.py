# Import the libraries
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Paths to the stop words and sentiment dictionaries
POSITIVE_WORDS_FILE = 'positive-words.txt'
NEGATIVE_WORDS_FILE = 'negative-words.txt'
STOPWORDS_FILES = [
    'StopWords_Generic.txt',
    'StopWords_Auditor.txt',
    'StopWords_Currencies.txt',
    'StopWords_DatesandNumbers.txt',
    'StopWords_GenericLong.txt',
    'StopWords_Geographic.txt',
    'StopWords_Names.txt'
]
OUTPUT_FILE = 'Output.csv'

def load_stop_words(files):
    """Loads stop words from multiple files."""
    stopwords = set()
    for file_path in files:
        with open(file_path, 'r') as file:
            stopwords.update(word.strip().lower() for word in file.readlines())
    return stopwords

def load_sentiment_words(file_path, stopwords):
    """Loads sentiment words, excluding any stopwords."""
    with open(file_path, 'r') as file:
        return {word.strip(): 1 if 'positive' in file_path else -1 for word in file if word.strip().lower() not in stopwords}

def extract_article_text(url):
    """Retrieves article text from URL using BeautifulSoup"""
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        # Define possible classes or structures for content
        possible_content_classes = [
            'td-post-content tagdiv-type'
        ]

        # Find the main content of the article
        article_content = None
        for class_name in possible_content_classes:
            article_content = soup.find('div', class_=class_name)
            if article_content:
                break

        # If specified classes are not found, extract text while excluding header or footer
        if not article_content:
            header = soup.find('header')
            footer = soup.find('footer')

            # Exclude header and footer content
            if header:
                header.extract()
            if footer:
                footer.extract()

            article_content = soup.body  

        # Extract text from the article content
        text = ""
        for tag in article_content.find_all(['p', 'li']):
            # Exclude any unwanted tags or classes here if necessary
            text += tag.get_text() + " "

        title = soup.find('title').text  
        return title, text.strip()

    except requests.HTTPError as e:
        print(f"Error fetching article from {url}: {e}")
        return None, None


def count_syllables(word):
    """Counts syllables in a word."""
    word = word.lower()
    vowels = "aeiouy"
    count = sum(1 for letter in word if letter in vowels)
    count -= sum(1 for ending in ['es', 'ed'] if word.endswith(ending))
    count += 1 if word.endswith('le') else 0
    count = max(1, count)
    return count

def calculate_text_metrics(text, stopwords, positive_dict, negative_dict):
    """Calculates text metrics."""
     # Preprocess text: normalize spaces, remove non-alphanumeric characters (except spaces and punctuation)
    processed_text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    processed_text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', processed_text)  # Remove non-alphanumeric/essential punctuation
    
    # Tokenize sentences and words
    sentences = sent_tokenize(processed_text)
    words = word_tokenize(processed_text)
    words = [word for word in words if word.lower() not in stopwords and word.isalpha()]
    positive_score = sum(1 for word in words if word in positive_dict)
    negative_score = sum(1 for word in words if word in negative_dict)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    sentences = sent_tokenize(text)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    complex_words = [word for word in words if count_syllables(word) > 2]
    percentage_of_complex_words = len(complex_words) / len(words) if words else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    syllable_count = sum(count_syllables(word) for word in words)
    personal_pronouns = len(re.findall(r'\b(I|we|us|our|ours|my|mine)\b', text, re.I))
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
        'COMPLEX WORD COUNT': len(complex_words),
        'WORD COUNT': len(words),
        'SYLLABLE PER WORD': syllable_count / len(words) if words else 0,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

def main():
    stopwords = load_stop_words(STOPWORDS_FILES)
    positive_dict = load_sentiment_words(POSITIVE_WORDS_FILE, stopwords)
    negative_dict = load_sentiment_words(NEGATIVE_WORDS_FILE, stopwords)
    data = pd.read_excel('Input.xlsx')
    output_columns = ['URL_ID', 'URL'] + list(calculate_text_metrics("test", stopwords, positive_dict, negative_dict).keys())
    output_df = pd.DataFrame(columns=output_columns)

    for _, row in data.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        print(f"Processing URL: {url}")
        title, text = extract_article_text(url)
        if not text:
            print(f"Skipping URL due to fetch error: {url}")
            continue
        metrics = calculate_text_metrics(text, stopwords, positive_dict, negative_dict)
        output_row = [url_id, url] + list(metrics.values())
        output_df = output_df._append(pd.Series(output_row, index=output_columns), ignore_index=True)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print("All URLs processed.")

if __name__ == "__main__":
    main()
