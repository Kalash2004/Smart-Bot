import speech_recognition as sr
import win32com.client
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from RAG_chain import ask_query

# Model initialization
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModelForCausalLM.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


lemmatizer = WordNetLemmatizer()

with open("Smartbot/intents.json", "r") as file:
    intents = json.load(file)

voice_intents = [
    "open google", "play music", "send email", "open youtube",
    "what is the time", "open instagram", "stop"
]

words = pickle.load(open(r'C:\Users\Kalash Srivastava\Demo-Projects\Smartbot\words.pkl', 'rb'))
classes = pickle.load(open(r'C:\Users\Kalash Srivastava\Demo-Projects\Smartbot\classes.pkl', 'rb'))
model = load_model(r'C:\Users\Kalash Srivastava\Demo-Projects\Smartbot\chatbot_model.h5')


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


print("Starting chat... Type 'exit' or 'stop' to end the conversation.")


chat_history_ids = None


generic_repsonse = [
    "I'm sorry ,I didn't understand that.",
    "Could you please rephrase that ?",
    "I don't know about that.",
    "I'm sorry, I don't know about that"
]


while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "stop", "quit"]:
        say("Alright! If you need anything feel free to ask me. Have a great day!")
        break
    
    if not user_input: 
        continue
    
    try:
        
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors='pt'
        )
        
        
        new_attention_mask = torch.ones_like(new_user_input_ids)
        
        
        if chat_history_ids is not None:
        
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            
           
            chat_length = chat_history_ids.shape[-1]
            new_length = new_user_input_ids.shape[-1]
            total_length = chat_length + new_length
            
            attention_mask = torch.ones((1, total_length), dtype=torch.long)
        else:
           
            bot_input_ids = new_user_input_ids
            attention_mask = new_attention_mask
       
        with torch.no_grad():
            chat_history_ids = transformer_model.generate(
                bot_input_ids,
                attention_mask=attention_mask, 
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.75,
                no_repeat_ngram_size=2 
            )
        
    
        response_start_idx = bot_input_ids.shape[-1]
        response_tokens = chat_history_ids[:, response_start_idx:]
        if output.strip() in generic_repsonse or  len(output.strip())<10:
            try:
                rag_answer = ask_query(user_input)
                if rag_answer and rag_answer.strip():
                    output = rag_answer
            except Exception as e:
                print("Error calling RAG")
            
        print(f"Bot :{output}")
        say(output)
        
        if response_tokens.shape[-1] == 0:
            output = "I'm not sure how to respond to that."
        else:
            output = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
            if not output.strip():  
                output = "Could you please rephrase that?"
        

        sentiment = analyze_sentiment(user_input)
        print(f"Bot ({sentiment}): {output}")
        
       
        if chat_history_ids.shape[-1] > 800:  
            chat_history_ids = chat_history_ids[:, -800:]
            
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Bot: I encountered an error. Let's start fresh.")
        chat_history_ids = None  
        continue
