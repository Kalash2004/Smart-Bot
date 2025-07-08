import speech_recognition as sr
import win32com.client
import webbrowser
import datetime
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


lemmatizer = WordNetLemmatizer()

with open(r"C:\Users\Kalash Srivastava\Projects\Smartbot\intents.json", "r") as file:
    intents = json.load(file)

voice_intents = [
    "open google", "play music", "send email", "open youtube",
    "what is the time", "open instagram", "open netflix",
    "open camera", "stop"
]

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])


def say(text):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    voices = speaker.GetVoices()
    
    speaker.Voice = voices[1]  
    speaker.Speak(text)


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.6
        print("Listening...")
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("Network error.")
        return "None"


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


def choose_mode():
    say("Do you want to interact with a Chatbot or Voice Assistant?")
    print("Say 'chat' for Chatbot or 'voice' for Voice Assistant:")

    while True:
        mode = takeCommand().lower()
        if "chat" in mode:
            say("You chose Chatbot mode.")
            print("You chose Chatbot mode.")
            return "chatbot"
        elif "voice" in mode:
            say("You chose Voice Assistant mode.")
            print("You chose Voice Assistant mode.")
            return "voice_assistant"
        else:
            say("Please say 'chat' for Chatbot or 'voice' for Voice Assistant.")
            print("Please say 'chat' or 'voice'.")


mode = choose_mode()

while True:
    if mode == "chatbot":
        user_input = input("You: ")
    else:
        user_input = takeCommand().lower()

    if "stop" in user_input or "exit" in user_input:
        say("Alright! If you need anything in the future, feel free to ask. Have a great day!")
        break

    sentiment = analyze_sentiment(user_input)
    print(f"(Sentiment: {sentiment})")

    if mode == "voice_assistant":
        if "google" in user_input:
            say("What do you want me to search on Google?")
            query = takeCommand().lower()
            webbrowser.open(f'https://www.google.com/search?q={query}')
        elif "time" in user_input:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            say(f"The current time is {current_time}")
        else:
            response = get_response(predict_class(user_input), intents)
            say(response)
            print(f"Bot({sentiment}): {response}")
    else:  
        response = get_response(predict_class(user_input), intents)
        print(f"Bot({sentiment}): {response}")
