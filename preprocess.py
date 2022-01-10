import re

from nltk import pos_tag
from nltk.corpus import wordnet

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')

# reading extended stop words
with open(r"helper_data\extend_stopwords.txt", "r") as file:
    extend_stopwords = file.read().split('\n')
    
with open(r"helper_data\pharasal_verbs.txt", 'r') as file:
    pharasal_verbs = eval(file.read())
    
with open(r"helper_data\remove_words.txt", 'r') as file:
    remove_words = file.read().split('\n')
    
with open(r"helper_data\replacement.txt", 'r') as file:
    replacement = eval(file.read())   
  

# extending stopwords
stop_words = stopwords.words('english')
stop_words.extend(extend_stopwords)
     
def remove_html_tags(text):
    html_tags = re.compile('<.*?>') 
    text = re.sub(html_tags, '', text)
    return text
    
def remove_urls(text):
    text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '', text)
    return text

def remove_emails(text):
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    return text

def remove_nonword_chars(text):
    return re.sub(r'[\W_]+', ' ', text)

def remove_emails(text):
    return re.sub(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)', 'email', text)

def remove_word_contains_numeric(text):
    return re.sub(r'\w*\d\w*', ' ', text)

def remove_nonword_chars(text):
    return re.sub(r'[\W_]+', ' ', text)

def remove_stopwords(text):
    text = [word for word in text.split(' ') if word not in stop_words]
    return ' '.join(text)

def remove_predefined_words(text):
    text = [word for word in text.split(' ') if word not in remove_words]
    return ' '.join(text)

def replace_predefined_words(text):
    text = [replacement.get(word, word) for word in text.split(' ')]
    return ' '.join(text)

def stem_words(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text.split(' ')]
    return ' '.join(text)

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
       
    try:
        text = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tag(text.split(' '))]
    except:
        return text
    return ' '.join(text)

def check_len(text):
    words = text.split(' ')
    words = [word for word in words if len(word) < 25 and len(word) >= 2]
    text = ' '.join(words)
    return text

def preprocess_text(text, word_root_preprocesser='lemmatizer'):
    
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_word_contains_numeric(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = replace_predefined_words(text)
    text = remove_predefined_words(text)
    
    text = remove_nonword_chars(text)
    text = remove_stopwords(text)
    
    if word_root_preprocesser == 'lemmatizer':
        text = lemmatize_words(text)
    elif word_root_preprocesser == 'stemmer':  
        text = stem_words(text)
      
    text = check_len(text)
    
    return text