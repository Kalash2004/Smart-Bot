
# 🤖 Description

SmartBot is an advanced AI-powered assistant that combines the capabilities of a traditional chatbot, modern transformer-based conversational AI, and voice interaction features. Built using machine learning, Hugging Face transformers, and natural language processing, SmartBot can understand queries, analyze sentiment, and deliver intelligent responses via text or voice.

With Retrieval-Augmented Generation (RAG) integration, SmartBot can retrieve factual answers from a connected knowledge base when the default conversational model lacks information, making it both engaging and informative.
# ✨ Features
### 🎯 Core Capabilities

Dual Interface – Choose between chatbot (text) or voice assistant mode.

Voice Recognition – Speech-to-text conversion for hands-free interaction.

Text-to-Speech – Natural voice responses using Windows SAPI.

Sentiment Analysis – Real-time emotion detection in user messages.

Intent Recognition – TensorFlow/Keras-powered classification for pre-defined commands.

RAG Integration – Retrieves and integrates factual answers from external sources.



### 🧠 Intelligence Features

Machine Learning – TensorFlow/Keras model for intent classification.

Transformer-based NLP – Hugging Face’s DialoGPT for contextual dialogue.

PyTorch 2.6 Ready – Compatibility fixes for token padding & attention masks.

Lemmatization & Tokenization – NLTK preprocessing for better understanding.

Fallback RAG Search – Uses ask_query() when transformer response is weak.

### 🎨 User Experience

Interactive Mode Selection – Voice-guided mode choice.

Real-time Feedback – Displays sentiment analysis with responses.

Error Handling – Handles recognition, generation, and RAG errors gracefully.

Conversation Memory – Maintains history for contextual replies (auto-truncated for performance).

### Programming Language

Python 3.x - Core development language

### Machine Learning & NLP

TensorFlow/Keras – Intent classification

Hugging Face Transformers – DialoGPT conversational model

PyTorch – Transformer execution backend

NLTK – Text preprocessing & lemmatization

TextBlob – Sentiment analysis

NumPy – Numerical computing

### Voice & Speech

SpeechRecognition – Speech-to-text

win32com.client – Windows SAPI text-to-speech

Google Speech Recognition – Cloud-based voice input

### Data & Web

JSON – Intent and response storage

Pickle – Model & data serialization

RAG – Retrieval-Augmented Generation for external knowledge

### Utilities

- datetime - Time-related functionalities
- random - Response randomization
- os/sys - System interactions# 🚀 Installation & Setup
### Prerequisites

Python 3.7+

Windows OS (for voice features)

Microphone (for voice mode)

Internet connection (for speech recognition & RAG)

### Installation Steps

Clone the repository
```bash
bashgit clone https://github.com/yourusername/smartbot.git
cd smartbot
```

### Install required dependencies
```bash
bashpip install speech-recognition
pip install pywin32
pip install tensorflow
pip install nltk
pip install textblob
pip install numpy


Download NLTK data
pythonimport nltk
nltk.download('punkt')
nltk.download('wordnet')

git clone https://github.com/yourusername/smartbot.git
cd smartbot
pip install -r requirements.txt
```
### Prepare model files

Ensure chatbot_model.h5 is in the project directory

Ensure words.pkl and classes.pkl are available

Update the path to intents.json in the code


Run the application
```bash
bashpython SmartBot.py
```

📋 Required Files

Make sure these files are present in your project directory:

Smartbot

├── SmartBot.py         # Main application file

├── RAG_chain.py        # Retrieval-Augmented Generation logic

├── intents.json        # Intent patterns and responses

├── chatbot_model.h5    # Trained TensorFlow/Keras model

├── words.pkl           # Vocabulary pickle file

├── classes.pkl         # Intent classes pickle file

└── README.md           # Project documentation

# 🔧 How It Works
1. Mode Selection

Upon startup, SmartBot asks the user to choose between chatbot or voice assistant mode
Voice commands are processed to determine the preferred interaction method

2. Input Processing

Chatbot Mode: Accepts text input from the user
Voice Assistant Mode: Uses speech recognition to convert voice to text

3. Intent Recognition

User input is tokenized and lemmatized using NLTK
A bag-of-words model converts text to numerical representation
TensorFlow model predicts the intent with confidence scores
Only predictions above the error threshold (0.25) are considered

4. Transformer Response

Generate conversational reply using DialoGPT.

5. RAG Fallback

 If reply is weak or generic, query external knowledge base.

7. Response Generation

Based on the predicted intent, a random response is selected from the JSON file
Responses are contextually appropriate and varied

7. Sentiment Analysis

TextBlob analyzes the emotional tone of user input
Sentiment is classified as positive, negative, or neutral
Results are displayed alongside bot responses

6. Voice Output

In voice assistant mode, responses are spoken using Windows SAPI
Female voice is selected by default for better user experience

7. Special Commands

Google Search: "open google" triggers search functionality
Time Query: "what is the time" provides current time
Exit Commands: "stop" or "exit" terminates the session
# Contributing

Contributions are always welcome!

If you'd like to contribute to this project, feel free to open an issue or submit a pull request. Contributions are welcome!

🤝 Contributing
We welcome contributions to improve SmartBot! Here's how you can help:
Ways to Contribute

- Bug Reports: Report issues or bugs you encounter
- Feature Requests: Suggest new features or improvements
- Code Contributions: Submit pull requests with enhancements
- Documentation: Help improve documentation and examples
- Testing: Test the bot on different systems and scenarios

Getting Started

Fork the repository
Create a feature branch
```bash
bashgit checkout -b feature/your-feature-name
```
Make your changes
Test thoroughly
Submit a pull request

Contribution Guidelines

Follow Python PEP 8 style guidelines,
Add comments for complex logic,
Update documentation for new features,
Test your changes before submitting,
Ensure compatibility with existing functionality.

Areas for Improvement

Cross-platform compatibility (Linux/Mac support),
Additional voice engines (Amazon Polly, Google TTS),
More intent categories (weather, news, calculations),
Database integration for conversation history,
GUI interface for better user experience,
Multi-language support,
Voice training capabilities.
