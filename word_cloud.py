import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import cv2 

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Provided text
text = """
Artificial Intelligence (AI) is rapidly transforming the world, 
with recent developments focusing on making AI systems more powerful 
and autonomous. Major AI companies are increasingly discussing the 
need for international cooperation and safeguards as AI becomes more 
capable. AI can now perform complex tasks such as writing code, 
conducting research, analyzing information, and interacting with 
digital systems. It is being used in healthcare, education, finance, 
software development, and scientific research. However, its rapid 
development has also raised concerns about privacy, security, 
misinformation, employment, and human control. Researchers and 
policymakers are therefore working to encourage AI innovation while
 ensuring that these systems are developed and used responsibly.
"""
print(text)

# Tokenization of words
words = word_tokenize(text.lower())  # Convert to lowercase for uniformity
print(words)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
# The general syntax for a list comprehension is: [expression for item in iterable if condition]
print(filtered_words)

# Create a WordCloud
word_freq = Counter(filtered_words)
print(word_freq)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()

cv2.imwrite(wordcloud, r'C:\Users\HP\Downloads') 