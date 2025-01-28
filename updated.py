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
nltk.download('punkt')

# Reading the Q&A corpus
def load_corpus(filepath):
    """Load the Q&A pairs from a formatted file using 'Q:' as a delimiter for questions."""
    with open(filepath, 'r', encoding='utf8', errors='ignore') as fin:
        raw = fin.read().strip()
    
    qa_pairs = {}
    lines = raw.split('\n')  # Split the file by lines
    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):  # If a new question starts
            if current_question:  # Save the previous question and answer
                qa_pairs[current_question] = ' '.join(current_answer).strip()
            current_question = line[2:].strip()  # Extract the question text
            current_answer = []  # Reset the answer list
        elif line.startswith("A:") or current_question:  # Collect the answer
            current_answer.append(line if not line.startswith("A:") else line[2:].strip())
    
    # Save the last question and answer
    if current_question:
        qa_pairs[current_question] = ' '.join(current_answer).strip()
    
    return qa_pairs

# Load the Q&A pairs
qa_data = load_corpus('UIUDATA.txt')

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
        robo_response = "I am sorry! I don't understand you. Can you teach me?"
        return robo_response, True  # Return True to indicate that teaching is possible
    else:
        # Handle line breaks: Replace '-' with '\n' to display on new lines
        robo_response = qa_data[questions[idx]].replace('\\n', '\n')  # Interpret line breaks
        robo_response = robo_response.replace('-', '\n-')  # Replace '-' with newlines for better formatting
        return robo_response, False


# Train the bot with new Q&A
def train_bot(user_question, user_answer):
    """Add a new Q&A pair to the corpus and make it immediately available."""
    # Convert multi-line answers into a single line with '\\n' for newlines
    formatted_answer = user_answer.replace('\n', '\\n')  # Replace actual newlines with '\\n'
    qa_data[user_question] = formatted_answer  # Store in the dictionary
    questions.append(user_question)  # Add the question to the questions list
    with open('UIUDATA.txt', 'a', encoding='utf8') as file:
        # Write in the format: Q: question\nA: answer
        file.write(f"Q: {user_question}\nA: {formatted_answer}\n")
    print("Thank you for teaching me!")

# Main interaction loop
flag = True
print("My name is Nexora. I will answer your queries. If you want to exit, type Bye!")
while flag:
    user_response = input("You: ").lower()
    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            print("Nexora: You are welcome.")
        else:
            if greeting(user_response) is not None:
                print("Nexora: " + greeting(user_response))
            else:
                response_text, teach_flag = response(user_response)
                print("Nexora: " + response_text)
                if teach_flag:  # If the bot didn't understand, ask for teaching
                    teach = input("Would you like to teach me? (yes/no): ").lower()
                    if teach == 'yes':
                        user_answer = input("Please provide the correct answer: ")
                        train_bot(user_response, user_answer)
                    else:
                        print("Nexora: Okay, skipping this for now.")
    else:
        flag = False
        print("Nexora: Bye! Take care.")
