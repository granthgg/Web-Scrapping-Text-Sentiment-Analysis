# TEXT EXTRACTION AND ANALYSIS

The project aims to automate the extraction of textual data from specified URLs and conduct a comprehensive analysis to calculate metrics related to sentiment, readability, and textual complexity.

## Approach to the Solution

1. **Setting Up the Environment:**
   - Ensure Python is installed on your system.
   - Install necessary Python libraries: beautifulsoup4, requests, pandas, and nltk, essential for web scraping, data manipulation, and natural language processing.

2. **Data Extraction:**
   - Utilizes BeautifulSoup and requests for fetching and parsing HTML, extracting main article text.
   - Custom function `extract_article_text` navigates DOM to isolate textual content, focusing on relevant elements while excluding headers and footers.

3. **Text Analysis:**
   - Employs nltk for tokenizing text into sentences and words, crucial for metric calculations.
   - Filters out stop words using compiled lists to refine analysis.
   - Computes sentiment scores using predefined lists of positive and negative words against filtered text.

4. **Metric Calculation:**
   - Calculates sentiment scores, readability scores (Fog Index), average sentence length, word count, and other metrics.
   - Defines custom functions for tasks like syllable counting to determine text complexity.

5. **Output Compilation:**
   - Compiles metrics into a pandas DataFrame, structured per output requirements.
   - Exports DataFrame as Output.csv, facilitating easy analysis.

## Instructions for Running the Script

 **Dependencies:** Install all dependencies using pip:
   ```shell
   pip install beautifulsoup4 requests pandas nltk
