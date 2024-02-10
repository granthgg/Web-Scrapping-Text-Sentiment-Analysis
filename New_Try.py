import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

STOPWORDS_DIR = 'StopWords'
OUTPUT_FILE = 'Output.csv'

def load_word_lists(directory):
    """Loads stop words and sentiment dictionaries"""
    stopwords = set()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            stopwords.update(f.read().splitlines())

    with open('positive-words.txt', 'r') as f:
        positive_dict = {word.strip(): 1 for word in f if word.strip() not in stopwords}
    with open('negative-words.txt', 'r') as f:
        negative_dict = {word.strip(): -1 for word in f if word.strip() not in stopwords}

    return stopwords, positive_dict, negative_dict

def extract_article_text(url):
    """Retrieves article text from URL using BeautifulSoup"""
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        # Define possible classes or structures for main content
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
    """Implements syllable counting based on your analysis rules"""
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

def calculate_text_metrics(text):
    """Implements all metric calculations as defined in your document"""
    cleaned_text = ' '.join(word for word in text.split() if word not in stopwords)


    # Sentiment Calculations
    positive_score = sum(1 for word in cleaned_text.split() if word in positive_dict)
    negative_score = sum(1 for word in cleaned_text.split() if word in negative_dict)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / ((len(cleaned_text.split())) + 0.000001)

    sentences = nltk.sent_tokenize(cleaned_text)
    words = nltk.word_tokenize(cleaned_text)

    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
    complex_words = [w for w in words if count_syllables(w) > 2]
    percentage_of_complex_words = len(complex_words) / len(words) 
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)

    complex_word_count = len(complex_words)
    word_count = len(words) 
    syllable_per_word = sum(count_syllables(w) for w in words) / len(words)
    personal_pronouns = len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', cleaned_text, re.I))
    avg_word_length = sum(len(w) for w in words) / len(words)

    # Return all the calculated results
    return positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length, \
           percentage_of_complex_words, fog_index, avg_sentence_length, complex_word_count, word_count, \
           syllable_per_word, personal_pronouns, avg_word_length
           

def save_text_to_file(url_id, url, text):
    """Saves extracted text to a file in the 'extracted_text' folder"""
    folder_name = 'Extracted Text'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    filename = os.path.join(folder_name, f"Text_{url_id}.txt")
    with open(filename, 'w', encoding='utf-8') as f:  
        f.write(f"URL ID: {url_id}\n")
        f.write(f"URL: {url}\n\n")
        f.write(text)
        
# Main Execution 
if __name__ == "__main__":
    stopwords, positive_dict, negative_dict = load_word_lists(STOPWORDS_DIR)
    data = pd.read_excel('Input.xlsx')

    # Load existing output
    if os.path.exists(OUTPUT_FILE):
        output_df = pd.read_csv(OUTPUT_FILE)
    else:
        output_columns = [  
            'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',	'SUBJECTIVITY SCORE',	
            'AVG SENTENCE LENGTH',	'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
            'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
        ]
        output_df = pd.DataFrame(columns=output_columns)

    for _, row in data.iterrows():
        url_id = row['URL_ID']
        url = row['URL']

        if url in output_df['URL'].values:
            print(f"Skipping already processed URL: {url}")
            continue

        print(f"Processing URL: {url}")
        title, text = extract_article_text(url)
        if title is None and text is None:
            print(f"Skipping URL due to HTTP error: {url}")
            continue

        # Save extracted text to file
        save_text_to_file(url_id, url, text)

        metrics = calculate_text_metrics(text)

        # Check if metrics has the correct number of elements
        if len(metrics) != len(output_columns) - 2:  
            print(f"Skipping URL due to incorrect number of metrics: {url}")
            continue

        output_row = [url_id, url, *metrics]  
        output_df = output_df._append(pd.Series(output_row, index=output_columns), ignore_index=True)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print("All URLs processed.")
