import string
import re
import nltk
from nltk.corpus import stopwords
from langchain.text_splitter import NLTKTextSplitter

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    cleaned_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return cleaned_text

def remove_special_characters(text):
    cleaned_text = "".join([char for char in text if char not in string.punctuation])
    return cleaned_text

def remove_html_tags(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    return cleaned_text

def clean_text(text):
    return remove_html_tags(remove_stopwords(text))

text = "Cats are known for their independent nature and playful antics, often entertaining their owners with their curious explorations and sudden bursts of energy. They are also incredibly affectionate creatures, \
seeking out cuddles and attention when they feel the need for companionship. With their soft fur and expressive eyes, cats have a unique charm that has captivated humans for centuries."

cleaned_text = clean_text(text)

print(f"original text:\n{text} \n\n")
print(f"cleaned text:\n{cleaned_text} \n\n")

text_splitter = NLTKTextSplitter()

sentences = text_splitter.split_text(cleaned_text)

print("text split: \n")

for sentence in sentences:
  print(sentence)