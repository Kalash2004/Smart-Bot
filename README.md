
# 🤖 Description

SmartBot is a sophisticated AI-powered assistant that combines the capabilities of a traditional chatbot with voice interaction features. Built using machine learning and natural language processing, SmartBot can understand user queries, analyze sentiment, and provide intelligent responses through both text and voice interfaces. The bot is designed to be user-friendly and can perform various tasks including web searches, time queries, and general conversation.
# ✨ Features
### 🎯 Core Capabilities

Dual Interface: Choose between chatbot (text) or voice assistant mode
Voice Recognition: Speech-to-text conversion for hands-free interaction
Text-to-Speech: Natural voice responses using Windows SAPI
Sentiment Analysis: Real-time emotion detection in user messages
Intent Recognition: Advanced NLP model to understand user intentions
Web Integration: Direct Google search functionality
Time Queries: Current time retrieval and announcement

### 🧠 Intelligence Features

Machine Learning: TensorFlow/Keras-based intent classification
Natural Language Processing: NLTK-powered text processing
Lemmatization: Advanced word normalization for better understanding
Probability-based Responses: Confidence scoring for intent predictions
Customizable Responses: JSON-based intent and response management

### 🎨 User Experience

Interactive Mode Selection: Voice-guided mode selection
Real-time Feedback: Live sentiment analysis display
Error Handling: Graceful handling of speech recognition errors
Exit Commands: Multiple ways to terminate the session# Tech Stack

### Programming Language

Python 3.x - Core development language

### Machine Learning & NLP

- TensorFlow/Keras - Deep learning model for intent classification
- NLTK - Natural language processing toolkit
- TextBlob - Sentiment analysis and text processing
- NumPy - Numerical computing for model operations

### Voice & Speech

- SpeechRecognition - Voice input processing
- win32com.client - Windows Speech API integration
- Google Speech Recognition - Cloud-based speech-to-text

### Data & Web

- JSON - Intent and response data management
- Pickle - Model and data serialization
- webbrowser - Web integration for searches

### Utilities

- datetime - Time-related functionalities
- random - Response randomization
- os/sys - System interactions# 🚀 Installation & Setup
### Prerequisites

- Python 3.7 or higher
- Windows OS (for voice features)
- Microphone (for voice assistant mode)
- Internet connection (for speech recognition)

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

smartbot/
├── SmartBot.py          # Main application file
├── intents.json         # Intent patterns and responses
├── chatbot_model.h5     # Trained TensorFlow model
├── words.pkl           # Vocabulary pickle file
├── classes.pkl         # Intent classes pickle file
└── README.md           # This file
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

4. Response Generation

Based on the predicted intent, a random response is selected from the JSON file
Responses are contextually appropriate and varied

5. Sentiment Analysis

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
