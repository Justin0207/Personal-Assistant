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


opening_text = [
    "Cool, I'm on it sir.",
    "Okay sir, I'm working on it.",
    "Just a second sir.",
]


USERNAME = 'Chima'


def SpeakText(command):

    # Initialize the engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    #voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-GB_HAZEL_11.0"

    engine.runAndWait()


def greet_user():
    hour = datetime.now().hour
    if (hour >= 0) and (hour < 12):
        SpeakText('Good Morning {}'.format(USERNAME))

    elif (hour >= 12) and (hour < 16):
        SpeakText('Good Afternoon {}'.format(USERNAME))

    elif (hour >= 16) and (hour < 20):
        SpeakText('Good Evening Mr {}'.format(USERNAME))
    SpeakText('I am travis. Your personal assistant. How may I assist you?')


listening = False


def start_listening():
    global listening
    listening = True
    print('Started Listening')


def pause_listening():
    global listening
    listening = False
    print('Stopped Listening')


keyboard.add_hotkey('ctrl+alt+k', start_listening)
keyboard.add_hotkey('ctrl+alt+p', pause_listening)


def take_user_input():
    r = sr.Recognizer()
    with sr.Microphone() as source2:
        #time.sleep(3)

        # wait for a second to let the recognizer
        # adjust the energy threshold based on
        # the surrounding noise level
        print('Listening......')
        # r.pause_threshold = 1
        # r.adjust_for_ambient_noise(source2, duration=1)
        # r.dynamic_energy_threshold = True

        # listens for the user's input
        audio2 = r.listen(source2)

        # Using google to recognize audio
    try:
        print('Recognizing...')
        query = r.recognize_google(audio2, language='en-NG')
        if 'exit' in query or 'stop' in query:
            #     SpeakText(choice(opening_text))
            # else:
            hour = datetime.now().hour
            if hour >= 21 and hour < 6:
                SpeakText("Good night sir, take care!")
            else:
                SpeakText('Have a good day sir!')
            exit()
    except Exception:
        SpeakText('Sorry, I could not understand. Could you please say that again?')
        query = 'None'
    return query


def search_on_wikipedia(query):
    results = wikipedia.summary(query, sentences=3)
    return results


def get_latest_news():
    news_headlines = []
    res = requests.get(
        f"https://newsapi.org/v2/top-headlines?country=ng&apiKey=031c3881f21d4fb3a4e4a4888091c81e&category=general").json()
    articles = res["articles"]
    for article in articles:
        news_headlines.append(article["title"])
    return news_headlines[:5]


def find_my_ip():
    ip_address = requests.get('https://api64.ipify.org?format=json').json()
    return ip_address["ip"]


def get_weather_report(city):
    res = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=f33975770181cc6ecd4126574bb64e5f&units=metric").json()
    weather = res["weather"][0]["main"]
    temperature = res["main"]["temp"]
    feels_like = res["main"]["feels_like"]
    #weather_description = res['main']["description"]
    return weather, f"{temperature}℃", f"{feels_like}℃"


def get_random_joke():
    url = "https://jokes-always.p.rapidapi.com/erJoke"

    headers = {
        "x-rapidapi-key": "852ba41666msh9aa06b7f475a963p1c4482jsnc618517f38f4",
        "x-rapidapi-host": "jokes-always.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers).json()

    return(response['data'])


def chat(query):


    url = "https://chatgpt-42.p.rapidapi.com/conversationgpt4-2"
    
    payload = {
    	"messages": [
    		{
    			"role": "user",
    			"content": query
    		}
    	],
    	"system_prompt": "",
    	"temperature": 0.9,
    	"top_k": 5,
    	"top_p": 0.9,
    	"max_tokens": 100,
    	"web_access": False
    }
    headers = {
    	"x-rapidapi-key": "852ba41666msh9aa06b7f475a963p1c4482jsnc618517f38f4",
    	"x-rapidapi-host": "chatgpt-42.p.rapidapi.com",
    	"Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers).json()


    reply = response['result']
    reply = reply.replace('OpenAI', 'Chima')

    return reply


def play_music(search_song):

    music_dir = r'C:/Users/HP/Music'  # Corrected the path string
    #search_song = 'swing low sweet chariot'

    # List all files in the specified music directory
    songs = os.listdir(music_dir)
    ratios = []
    # Iterate over all songs in the directory
    for song in songs:
        # Calculate the similarity ratio
        ratio = fuzz.partial_ratio(song, search_song)
        ratios.append(ratio)
        # If the similarity ratio is above a threshold (e.g., 0.6), play the song
    for i, scores in enumerate(ratios):
        if scores == max(ratios):
            os.startfile(os.path.join(music_dir, songs[i]))
            break  # Stop after finding the first matching song


if __name__ == '__main__':

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    # joyce_image = face_recognition.load_image_file(r"C:\Users\HP\Pictures\Camera Roll\WIN_20231117_01_52_49_Pro.jpg")
    # joyce_face_encoding = face_recognition.face_encodings(joyce_image)[0]

    # Load a second sample picture and learn how to recognize it.
    chima_image = face_recognition.load_image_file(
        r"C:\Users\HP\Pictures\Camera Roll\WIN_20240602_08_37_41_Pro.jpg")
    chima_face_encoding = face_recognition.face_encodings(chima_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        # joyce_face_encoding,
        chima_face_encoding
    ]
    known_face_names = [
        "Chima"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Display the resulting imageK
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if 'Chima' in face_names:
            greet_user()
            video_capture.release()  # Stop capturing from the webcam
            cv2.destroyAllWindows()  # Close any OpenCV windows
            break  # Exit the while loop to stop the program

    while True:

        if listening:

            query = take_user_input().lower()
            listening = False

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
                # listening = False

            elif ("open microsoft edge" in query) or ("open edge" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("microsoft edge")

            elif ("open excel" in query) or ("open msexcel" in query) or ("open sheet" in query) or ("open spreadsheet" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("excel")

            elif ("open slide" in query) or ("open mspowerpoint" in query) or ("open ppt" in query) or ("open powerpoint" in query):
                SpeakText(choice(opening_text))
                AppOpener.open("powerpoint")

            elif ("open word" in query) or ("open microsoft edge" in query):
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

            elif ("close excel" in query) or ("close msexcel" in query) or ("close sheet" in query) or ("close spreadsheet" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("excel")

            elif ("close slide" in query) or ("close mspowerpoint" in query) or ("close ppt" in query) or ("close powerpoint" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("powerpoint")

            elif ("close word" in query) or ("close microsoft edge" in query):
                SpeakText(choice(opening_text))
                AppOpener.close("word")

            elif 'the time' in query:
                strTime = datetime().now().strftime("%H:%M:%S")
                SpeakText(f"The time is {strTime}")

            elif 'wikipedia' in query:
                try:
                    SpeakText(
                        'What do you want to search on Wikipedia, sir?')
                    search_query = take_user_input().lower()
                    listening = False
                    SpeakText(choice(opening_text))
                    results = search_on_wikipedia(search_query)
                    SpeakText(f"According to Wikipedia, {results}")
                    SpeakText(
                        "For your convenience, I am printing it on the screen sir.")
                    print(results)
                except wikipedia.DisambiguationError:
                    SpeakText(
                        'Sorry, I could not understand. Could you please say that again?')
                    search_query = take_user_input().lower()
                    listening = False
                    SpeakText(choice(opening_text))
                    results = search_on_wikipedia(search_query)
                    SpeakText(f"According to Wikipedia, {results}")
                    SpeakText(
                        "For your convenience, I am printing it on the screen sir.")
                    print(results)

            elif ('news' in query) or ('latest news' in query):
                SpeakText(
                    "I'm reading out the latest news headlines, sir")
                SpeakText(get_latest_news())
                SpeakText(
                    "For your convenience, I am printing it on the screen sir.")
                print(*get_latest_news(), sep='\n')

            elif 'ip address' in query:
                ip_address = find_my_ip()
                SpeakText(
                    f'Your IP Address is {ip_address}.\n For your convenience, I am printing it on the screen sir.')
                print(f'Your IP Address is {ip_address}')

            elif ('open website' in query) or ('visit website' in query):
                SpeakText('What website do you want to visit, sir?')
                search_query = take_user_input().lower()
                listening = False
                SpeakText(choice(opening_text))
                chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
                webbrowser.register(
                    'chrome', None, webbrowser.BackgroundBrowser(chromePath))
                web = webbrowser.get('chrome')
                web.open_new(search_query)

            elif 'weather' in query:
                ip_address = find_my_ip()
                city = requests.get(
                    f"https://ipapi.co/{ip_address}/city/").text
                city = 'Benin city'
                SpeakText(
                    f"Getting weather report for your city {city}")
                weather, temperature, feels_like = get_weather_report(
                    city)
                SpeakText(
                    f"The current temperature is {temperature}, but it feels like {feels_like}")
                SpeakText(
                    f"Also, the weather report talks about {weather}")
                SpeakText(
                    "For your convenience, I am printing it on the screen sir.")
                print('----------WEATHER REPORT-------------')
                print(
                    f"Description: {weather}\nTemperature: {temperature}\nFeels like: {feels_like}")

            elif 'joke' in query:
                SpeakText(f"Hope you like this one sir")
                joke = get_random_joke()
                SpeakText(joke)
                SpeakText(
                    "For your convenience, I am printing it on the screen sir.")
                print(joke)

            elif ('play music' in query) or ("play song" in query):
                SpeakText("what song do you want to play?")
                listening = True
                search_song = take_user_input().lower()
                listening = False
                play_music(search_song)

            else:
                # SpeakText("what sir?")
                # search_query = take_user_input().lower()
                conversation = chat(query)
                SpeakText(conversation)
                SpeakText(
                    "For your convenience, I am printing it on the screen sir.")
                print(conversation)
    # Hit 'q' on the keyboard to quit!
    

# Release handle to the webcam

# video_capture.release()
# cv2.destroyAllWindows()
