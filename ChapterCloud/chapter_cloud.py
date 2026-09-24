import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# 1. Read the text file

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Tokenize

tokens = word_tokenize(text.lower())

# 3. Remove stopwords

stop_words = set(stopwords.words("english"))

words = []

for word in tokens:

    if word.isalpha() and word not in stop_words:
        words.append(word)


# Convert words back into text
clean_text = " ".join(words)



# 4. Generate WordCloud


wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color="white",
    max_words=150
).generate(clean_text)



# 5. Display WordCloud

plt.figure(figsize=(14, 8))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title("WordCloud")

plt.show()



# 6. Save WordCloud

wordcloud.to_file(
    "wordcloud.png"
)

print("WordCloud generated successfully!")
print("Saved as: wordcloud.png")