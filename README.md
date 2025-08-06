# Travis - Voice-Controlled AI Desktop Assistant

**Travis** is an intelligent, multimodal desktop assistant that uses **voice**, **face**, and **keyboard shortcuts** for control. It leverages **voiceprint recognition**, **facial recognition**, **NLP**, and a range of APIs to respond intelligently to your commands, offering an immersive experience with natural human interaction.

## ✨ Features

- 🔊 Voice recognition + speaker verification
- 🧠 NLP chatbot using ChatGPT API
- 🖼️ Facial recognition for user identification
- 🔍 Wikipedia search
- 📰 Latest Nigerian news
- 🌦️ Live weather reports
- 🎵 Local music player
- 🎭 Jokes on demand
- 📷 Screenshot capture
- 🕹️ App and browser launcher
- 🛑 Hotkey support (`Ctrl+Alt+K` to start, `Ctrl+Alt+P` to pause)
- 💻 System control (shutdown, sleep, restart)

---

## 🛠️ Setup Instructions

### Requirements
- Python 3.8+

  
### Dependencies include:

- pyttsx3
- speech_recognition
- wikipedia
- face_recognition
- opencv-python
- torch
- torchaudio
- speechbrain
- thefuzz
- sounddevice
- AppOpener
- requests



### Environment Setup
Set the following in your .env or via os.environ in the code:

RAPIDAPI_KEY – for jokes and ChatGPT APIs

Replace:

<<YOUR_EMBEDDING_PATH>> with your .npy voice embedding file

<<YOUR_NEWS_API_KEY>> – for NewsAPI.org

<<YOUR_WEATHER_API_KEY>> – for OpenWeatherMap

<<YOUR_FACE_IMAGE_PATH>> – image used for facial recognition

<<YOUR_MUSIC_DIRECTORY>> – music folder path



## ▶️ How It Works
🔐 Speaker Verification
When prompted, the assistant listens for a short audio and matches the voice against known embeddings using speechbrain/spkrec-ecapa-voxceleb.

😎 Facial Recognition
Uses face_recognition and OpenCV to validate your identity before starting.

🧠 Intelligent Response
Default queries are passed to ChatGPT via RapidAPI for natural language response.

💾 Save Speaker Embedding
You can generate and save your own voice embedding with thE voice_embeddings.py script
💡 Use a clean, 3–5 second clip of your voice for best results.

🎯 Hotkey Controls
Action	Shortcut
Start listening	Ctrl+Alt+K
Pause listening	Ctrl+Alt+P



## 🧠 Future Improvements
Add GUI using PyQt or Tkinter

Support for multiple users (embedding management)

Offline fallback (e.g., basic responses)

Expand chatbot memory or integrate local LLMs



## 🙏 Acknowledgments
SpeechBrain

OpenAI

RapidAPI

face_recognition



## 📄 License
This project is under the Apache License. See LICENSE for details.



## Author: Anyanwu Chima Justice
🗓️ Created: June 20, 2025
📍 Nigeria

