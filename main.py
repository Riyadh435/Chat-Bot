import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK data (if not already installed)
nltk.download('popular', quiet=True)
nltk.download('punkt_tab')

# Reading the Q&A corpus
def load_corpus(filepath):
    """Load the Q&A pairs from mychat.txt into a dictionary."""
    with open(filepath, 'r', encoding='utf8', errors='ignore') as fin:
        raw = fin.read().strip()
    qa_pairs = {}
    for line in raw.split('\n'):
        if line.startswith('Q: '):
            question = line[3:].strip()
        elif line.startswith('A: '):
            answer = line[3:].strip()
            qa_pairs[question] = answer
    return qa_pairs

# Load the Q&A pairs
qa_data = load_corpus('mychat.txt')

# Extract questions for similarity matching
questions = list(qa_data.keys())

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    """Generate a response by matching user input to the closest question."""
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(questions + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare user input with questions
    idx = vals.argsort()[0][-1]  # Index of the most similar question
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]  # Highest similarity score
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = qa_data[questions[idx]]
    return robo_response

# Main interaction loop
flag = True
print("My name is Nexora. I will answer your queries. If you want to exit, type Bye!")
while flag:
    user_response = input("You: " ).lower()
    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            print("Nexora: You are welcome.")
        else:
            if greeting(user_response) is not None:
                print("Nexora: " + greeting(user_response))
            else:
                print("Nexora: ", end="")
                print(response(user_response))
    else:
        flag = False
        print("Nexora: Bye! Take care.")