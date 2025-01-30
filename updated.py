import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML, clear_output
import time

warnings.filterwarnings('ignore')

nltk.download('popular', quiet=True)
nltk.download('punkt')

def load_corpus(filepath):
    """Load the Q&A pairs from a formatted file using 'Q:' as a delimiter for questions."""
    with open(filepath, 'r', encoding='utf8', errors='ignore') as fin:
        raw = fin.read().strip()

    qa_pairs = {}
    lines = raw.split('\n')
    current_question = None
    current_answer = []

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            if current_question:
                qa_pairs[current_question] = ' '.join(current_answer).strip()
            current_question = line[2:].strip()
            current_answer = []
        elif line.startswith("A:") or current_question:
            current_answer.append(line if not line.startswith("A:") else line[2:].strip())

    if current_question:
        qa_pairs[current_question] = ' '.join(current_answer).strip()

    return qa_pairs

qa_data = load_corpus('UIUDATA.txt')
questions = list(qa_data.keys())

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["Hi!", "Hey!", "Hello!", "Hi there!", "I'm glad to chat with you."]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    """Generate a response by matching user input to the closest question."""
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(questions + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf == 0:
        return "I don't understand! Would you like to teach me? If yes, please provide the correct answer. Otherwise, type 'no' to skip.", True
    else:
        return qa_data[questions[idx]].replace('\\n', '<br>'), False

def train_bot(user_question, user_answer):
    """Add a new Q&A pair to the corpus and make it immediately available."""
    formatted_answer = user_answer.replace('\n', '\\n')
    qa_data[user_question] = formatted_answer
    questions.append(user_question)

    with open('UIUDATA.txt', 'a', encoding='utf8') as file:
        file.write(f"Q: {user_question}\nA: {formatted_answer}\n")

    return "*Thank you! I have learned this now.* 😊"


chat_history = []
pending_teach = None  

def display_chat():
    """Displays chat history with an input box in Colab."""
    clear_output(wait=True)
    chat_html = f"""
    <style>
        .chat-container {{
            width: 60%;
            margin: auto;
            border-radius: 10px;
            background: white;
            padding: 15px;
            color: black;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }}
        .bot {{
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            width: fit-content;
            max-width: 70%;
        }}
        .user {{
            background-color: #ADD8E6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            text-align: right;
            width: fit-content;
            max-width: 70%;
            margin-left: auto;
        }}
        .input-box {{
            max-width: 750px;
            display: flex;
            
            margin-top: 10px;
            padding: 10px;
            background: #f5f5f5;
            border-top: 1px solid #ddd;
            margin: 10px auto;
        }}
        input {{
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }}
        button {{
            padding: 10px;
            border: none;
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 5px;
        }}
    </style>
    <div class='chat-container'>
        {"".join(chat_history)}
    </div>
    <div class="input-box">
        <input type="text" id="user_input" placeholder="Type your message here..." autofocus>
        <button onclick="send_message()">Send</button>
    </div>
    <script>
        function send_message() {{
            google.colab.kernel.invokeFunction('notebook.send_message', [document.getElementById("user_input").value], {{}})
            document.getElementById("user_input").value = "";
        }}
    </script>
    """
    display(HTML(chat_html))


def chat():
    global chat_history
    chat_history.append("<div class='bot'><b>Nexora:</b> Hello! I will answer your queries.</div>")
    display_chat()


def process_message(user_response):
    global chat_history, pending_teach

    chat_history.append(f"<div class='user'><b>You:</b> {user_response}</div>")

    if pending_teach:
        if user_response.lower() == "no":
            bot_response = "Okay! I will not learn this. 😊"
            pending_teach = None  
        else:
            bot_response = train_bot(pending_teach, user_response)
            pending_teach = None  
    elif user_response.lower() == 'bye':
        bot_response = "Goodbye! Take care. 😊"
    elif user_response.lower() in ('thanks', 'thank you'):
        bot_response = "You're welcome! 😊"
    else:
        if greeting(user_response) is not None:
            bot_response = greeting(user_response)
        else:
            bot_response, teach_flag = response(user_response)
            if teach_flag:
                pending_teach = user_response  

    chat_history.append(f"<div class='bot'><b>Nexora:</b> {bot_response}</div>")
    display_chat()

# Register callback for sending messages
from google.colab import output
output.register_callback('notebook.send_message', process_message)

# Start the chat
chat()
