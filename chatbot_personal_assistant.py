# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:37:35 2024

@author: Anyanwu Chima Justice
"""

import pyttsx3
from datetime import datetime
from random import choice
import speech_recognition as sr
import time
import wikipedia
import requests
import face_recognition
import cv2
import numpy as np
import webbrowser
import json
import AppOpener
import os
from thefuzz import fuzz
import keyboard
import torch
import torchaudio
import sounddevice as sd
from scipy.spatial.distance import cosine
from speechbrain.inference import EncoderClassifier
import ctypes
from PIL import ImageGrab


opening_text = [
    "Cool, I'm on it sir.",
    "Okay sir, I'm working on it.",
    "Just a second sir.",
]

USERNAME = "Chima"  # Your name or registered speaker name

# Load speaker recognition model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
).to(device)

# Load stored embeddings
stored_embeddings = {
    "Chima": np.load("<<YOUR_EMBEDDING_PATH>>")  # Pre-saved voice embedding path
}

def SpeakText(command):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    engine.runAndWait()

def greet_user():
    hour = datetime.now().hour
    if 0 <= hour < 12:
        SpeakText(f'Good Morning {USERNAME}')
    elif 12 <= hour < 16:
        SpeakText(f'Good Afternoon {USERNAME}')
    elif 16 <= hour < 20:
        SpeakText(f'Good Evening Mr {USERNAME}')
    SpeakText('I am Travis, your personal assistant. How may I assist you?')

# Hotkey control
listening = False
def start_listening():
    global listening
    if not listening:
        listening = True
        print('✅ Started Listening')

def pause_listening():
    global listening
    listening = False
    print('⏸️ Stopped Listening')

keyboard.add_hotkey('ctrl+alt+k', start_listening)
keyboard.add_hotkey('ctrl+alt+p', pause_listening)

def take_user_input(duration=3, threshold=0.75, sample_rate=16000, max_retries=3):
    retries = 0
    while retries < max_retries:
        print("🎙️ Listening for voice input...")

        recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()

        waveform = torch.tensor(recording.T, dtype=torch.float32).to(device)

        with torch.no_grad():
            embedding = classifier.encode_batch(waveform).squeeze().cpu().numpy()

        best_match = None
        best_score = float("inf")

        for name, stored_emb in stored_embeddings.items():
            score = cosine(embedding, stored_emb)
            if score < best_score:
                best_score = score
                best_match = name

        similarity = 1 - best_score
        if best_score > threshold:
            SpeakText("Access denied. Unauthorized speaker. Please try again.")
            print(f"❌ Unknown speaker (Similarity: {similarity:.2f})")
            retries += 1
            continue

        print(f"✅ Speaker identified as: {best_match} (Confidence: {similarity:.2f})")

        try:
            r = sr.Recognizer()
            audio_int16 = (recording * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_data = sr.AudioData(audio_bytes, sample_rate, 2)
            query = r.recognize_google(audio_data, language='en-NG')
            print(f"🗣️ You said: {query}")

            if 'exit' in query or 'stop' in query:
                hour = datetime.now().hour
                if 21 <= hour or hour < 6:
                    SpeakText("Good night sir, take care!")
                else:
                    SpeakText('Have a good day sir!')
                exit()

            return query

        except Exception as e:
            SpeakText("Sorry, I couldn't understand you. Please try again.")
            print("Recognition error:", e)
            return None

    SpeakText("Maximum attempts reached. Please try again later.")
    return None

def search_on_wikipedia(query):
    try:
        results = wikipedia.summary(query, sentences=3)
        return results
    except wikipedia.DisambiguationError:
        SpeakText('That term is ambiguous. Please be more specific.')
    except wikipedia.PageError:
        SpeakText("I couldn't find any page on Wikipedia for that topic.")
    except Exception as e:
        SpeakText('Something went wrong while searching Wikipedia.')
        print("Wikipedia Error:", e)
    return None


def get_latest_news():
    news_headlines = []
    try:
        res = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=ng&apiKey=<<YOUR_NEWS_API_KEY>>&category=general"
        )
        res.raise_for_status()
        articles = res.json().get("articles", [])
        for article in articles:
            news_headlines.append(article.get("title", "No Title"))
        return news_headlines[:5]
    except Exception as e:
        SpeakText("Unable to fetch the latest news.")
        print("News API Error:", e)
        return []


def find_my_ip():
    try:
        ip_address = requests.get('https://api64.ipify.org?format=json').json()
        return ip_address["ip"]
    except Exception as e:
        SpeakText("Unable to fetch your IP address.")
        print("IP Error:", e)
        return "Unavailable"


def get_weather_report(city):
    try:
        res = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=<<YOUR_WEATHER_API_KEY>>&units=metric"
        )
        res.raise_for_status()
        data = res.json()
        weather = data["weather"][0]["main"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return weather, f"{temperature}℃", f"{feels_like}℃"
    except Exception as e:
        SpeakText("Unable to fetch the weather information.")
        print("Weather API Error:", e)
        return "Unknown", "N/A", "N/A"


def get_random_joke():
    try:
        url = "https://jokes-always.p.rapidapi.com/erJoke"
        headers = {
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
            "x-rapidapi-host": "jokes-always.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get('data', "No joke found.")
    except Exception as e:
        SpeakText("I couldn't fetch a joke right now.")
        print("Joke API Error:", e)
        return "No joke available."


def chat(query):
    try:
        url = "https://chatgpt-42.p.rapidapi.com/conversationgpt4-2"
        payload = {
            "messages": [{"role": "user", "content": query}],
            "system_prompt": "",
            "temperature": 0.9,
            "top_k": 5,
            "top_p": 0.9,
            "max_tokens": 100,
            "web_access": False
        }
        headers = {
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
            "x-rapidapi-host": "chatgpt-42.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        reply = response.json().get('result', "I'm not sure how to respond to that.")
        return reply.replace('OpenAI', 'Chima')
    except Exception as e:
        SpeakText("I'm unable to connect to the chat service.")
        print("Chat API Error:", e)
        return "I'm having trouble processing your request right now."


def play_music(search_song):
    music_dir = r'<<YOUR_MUSIC_DIRECTORY>>'
    songs = os.listdir(music_dir)
    ratios = []
    for song in songs:
        ratio = fuzz.partial_ratio(song, search_song)
        ratios.append(ratio)
    for i, scores in enumerate(ratios):
        if scores == max(ratios):
            os.startfile(os.path.join(music_dir, songs[i]))
            break

def shutdown_system():
    os.system("shutdown /s /f /t 0")

def sleep_system():
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

def restart_system():
    os.system("shutdown /r /f /t 0")

def confirm_action(action):
    SpeakText(f"Are you sure you want to {action}? Please say 'yes' to confirm or 'no' to cancel.")
    user_response = take_user_input()
    if user_response and 'yes' in user_response.lower():
        return True
    elif user_response and 'no' in user_response.lower():
        SpeakText("Operation canceled.")
        return False
    else:
        SpeakText("Sorry, I didn't understand that. Cancelling operation.")
        return False

def take_screenshot():
    try:
        screenshot = ImageGrab.grab()
        screenshot.save("screenshot.png")
        screenshot.close()
        SpeakText("Screenshot taken and saved.")
    except Exception as e:
        SpeakText("Failed to take screenshot.")
        print("Screenshot Error:", e)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    chima_image = face_recognition.load_image_file(r"<<YOUR_FACE_IMAGE_PATH>>")
    chima_face_encoding = face_recognition.face_encodings(chima_image)[0]

    known_face_encodings = [chima_face_encoding]
    known_face_names = ["Chima"]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if 'Chima' in face_names:
            greet_user()
            video_capture.release()
            cv2.destroyAllWindows()
            break

    while True:
        if listening:
            query = take_user_input().lower()
            listening = False

            # Process user command (same logic continues here...)
            # ⬇️ KEEP existing command blocks ⬇️
            if ('open camera' in query) or ('open webcam' in query):
                SpeakText(choice(opening_text))
                AppOpener.open("camera")

            elif 'open notepad' in query:
                SpeakText(choice(opening_text))
                AppOpener.open("notepad")

            elif 'open spyder' in query:
                SpeakText(choice(opening_text))
                AppOpener.open("spyder")

            elif 'open calculator' in query:
                SpeakText(choice(opening_text))
                AppOpener.open("calculator")

            elif ("open google" in query) or ("open chrome" in query) or ("open browser" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("google chrome")

            elif ("open microsoft edge" in query) or ("open edge" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("microsoft edge")

            elif ("open excel" in query) or ("open sheet" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("excel")

            elif ("open powerpoint" in query) or ("open slide" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("powerpoint")

            elif ("open word" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("word")

            elif 'close camera' in query or 'close webcam' in query:
                SpeakText(choice(opening_text))
                AppOpener.close("camera")

            elif 'close notepad' in query:
                SpeakText(choice(opening_text))
                AppOpener.close("notepad")

            elif 'close spyder' in query:
                SpeakText(choice(opening_text))
                AppOpener.close("spyder")

            elif 'close calculator' in query:
                SpeakText(choice(opening_text))
                AppOpener.close("calculator")

            elif ("close google" in query) or ("close chrome" in query) or ("close browser" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("google chrome")

            elif ("close microsoft edge" in query) or ("close edge" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("microsoft edge")

            elif ("close excel" in query) or ("close sheet" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("excel")

            elif ("close powerpoint" in query) or ("close slide" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("powerpoint")

            elif ("close word" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("word")

            elif 'the time' in query:
                strTime = datetime.now().strftime("%H:%M:%S")
                SpeakText(f"The time is {strTime}")

            elif 'wikipedia' in query:
                try:
                    SpeakText('What do you want to search on Wikipedia, sir?')
                    search_query = take_user_input().lower()
                    listening = False
                    SpeakText(choice(opening_text))
                    results = search_on_wikipedia(search_query)
                    SpeakText(f"According to Wikipedia, {results}")
                    print(results)
                except wikipedia.DisambiguationError:
                    SpeakText('Sorry, that term is ambiguous. Please be more specific.')
                except Exception as e:
                    SpeakText('Something went wrong while searching Wikipedia.')
                    print(e)

            elif ('news' in query) or ('latest news' in query):
                SpeakText("I'm reading out the latest news headlines, sir")
                headlines = get_latest_news()
                for headline in headlines:
                    SpeakText(headline)
                print(*headlines, sep='\n')

            elif 'ip address' in query:
                ip_address = find_my_ip()
                SpeakText(f'Your IP Address is {ip_address}')
                print(f'Your IP Address is {ip_address}')

            elif ('open website' in query) or ('visit website' in query):
                SpeakText('What website do you want to visit, sir?')
                search_query = take_user_input().lower()
                if search_query:
                    SpeakText(choice(opening_text))
                    if not search_query.startswith('http'):
                        search_query = 'https://' + search_query
                    webbrowser.open(search_query)

            elif 'weather' in query:
                ip_address = find_my_ip()
                city = requests.get(f"https://ipapi.co/{ip_address}/city/").text
                SpeakText(f"Getting weather report for {city}")
                weather, temperature, feels_like = get_weather_report(city)
                SpeakText(f"Current temperature is {temperature}, feels like {feels_like}, and weather is {weather}")
                print(f"Weather: {weather}\nTemperature: {temperature}\nFeels like: {feels_like}")

            elif 'joke' in query:
                joke = get_random_joke()
                SpeakText("Hope you enjoy this one:")
                SpeakText(joke)
                print(joke)

            elif ('play music' in query) or ("play song" in query):
                SpeakText("Which song would you like to play?")
                search_song = take_user_input().lower()
                play_music(search_song)

            elif 'shut down' in query or 'shutdown' in query:
                if confirm_action("shut down the system"):
                    SpeakText("Shutting down now, sir.")
                    shutdown_system()

            elif 'restart' in query or 'reboot' in query:
                if confirm_action("restart the system"):
                    SpeakText("Restarting now, sir.")
                    restart_system()

            elif 'sleep' in query or 'rest' in query:
                if confirm_action("put the system to sleep"):
                    SpeakText("Putting the system to sleep, sir.")
                    sleep_system()

            elif 'screenshot' in query or 'capture screen' in query:
                if confirm_action("take a screenshot"):
                    SpeakText("Taking a screenshot, sir.")
                    take_screenshot()
                    print("Screenshot saved.")

            else:
                # Default to chatbot if no command matched
                conversation = chat(query)
                SpeakText(conversation)
                print(conversation)
