import random
import re
from collections import defaultdict
import speech_recognition as sr
import subprocess
import datetime
import webbrowser
import pyaudio
import argparse
import cmd
import nltk
import requests
import os
import time
import optparse
import re
import site
import random
import socket
import webbrowser
import subprocess
from subprocess import call
from bs4 import BeautifulSoup
import numpy as np
import datetime
import transformers
import wikipedia
import pywhatkit
import pyjokes
import platform
import tensorflow as tf
import numpy as np
import threading
from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment

print('''
                  JJJJJJJJJJJ     OOOOOOOOO        SSSSSSSSSSSSSSS HHHHHHHHH     HHHHHHHHH
                  J:::::::::J   OO:::::::::OO    SS:::::::::::::::SH:::::::H     H:::::::H
                  J:::::::::J OO:::::::::::::OO S:::::SSSSSS::::::SH:::::::H     H:::::::H
                  JJ:::::::JJO:::::::OOO:::::::OS:::::S     SSSSSSSHH::::::H     H::::::HH
                    J:::::J  O::::::O   O::::::OS:::::S              H:::::H     H:::::H
                    J:::::J  O:::::O     O:::::OS:::::S              H:::::H     H:::::H
                    J:::::J  O:::::O     O:::::O S::::SSSS           H::::::HHHHH::::::H
                    J:::::j  O:::::O     O:::::O  SS::::::SSSSS      H:::::::::::::::::H
                    J:::::J  O:::::O     O:::::O    SSS::::::::SS    H:::::::::::::::::H
        JJJJJJJ     J:::::J  O:::::O     O:::::O       SSSSSS::::S   H::::::HHHHH::::::H
        J:::::J     J:::::J  O:::::O     O:::::O            S:::::S  H:::::H     H:::::H
        J::::::J   J::::::J  O::::::O   O::::::O            S:::::S  H:::::H     H:::::H
        J:::::::JJJ:::::::J  O:::::::OOO:::::::OSSSSSSS     S:::::SHH::::::H     H::::::HH
         JJ:::::::::::::JJ    OO:::::::::::::OO S::::::SSSSSS:::::SH:::::::H     H:::::::H
           JJ:::::::::JJ        OO:::::::::OO   S:::::::::::::::SS H:::::::H     H:::::::H
             JJJJJJJJJ            OOOOOOOOO      SSSSSSSSSSSSSSS   HHHHHHHHH     HHHHHHHHH
''')

MASTER = "Blake"

def wishme():
    hour = int(datetime.datetime.now().hour)

    if hour>=0 and hour <12:
        speak("Good Morning "+ MASTER)
    elif hour>=12 and hour <18:
        speak("Good Afternoon "+ MASTER)
    else:
        speak("Good Evening "+ MASTER)


def speak(text, speed=2.5):
    tts = gTTS(text=text, lang='en')
    filename = "audio.mp3"
    tts.save(filename)  # Save the speech audio into a file
    playsound(filename)  # Use playsound to play the saved mp3 file
    os.remove(filename)  # Remove the file after playing it

#put chatbot ai here
#class SimpleLLM:
#def __init__(self):
print("----------------------------")
print("----- Starting up Josh -----")
print("----------------------------")
wishme()


def handle_intent(intent, app_name=None):

    if intent == "open_app":
        if app_name:
            return open_app(app_name)
        else:
            return "Please specify an application to open."

    else:
        return "I'm sorry, I don't understand."


def open_app(app_name):
    try:
        subprocess.Popen(["open", "-a", app_name])
        return f"Opening {app_name}..."
    except FileNotFoundError:
        return f"Sorry, I could not find the {app_name} application."


def action_time():
    time = datetime.datetime.now().strftime("%H:%M")
    speak(f"The current Time is {time}")
    return f"The current Time is {time}"


def getthenews():
    url = ('https://www.bbc.com/news')
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')
    headlinesnews = soup.find('body').find_all('h3')
    unwantedbs = ['BBC World News TV', 'BBC World Service Radio',
                    'News daily newsletter', 'Mobile app', 'Get in touch']

    for x in list(dict.fromkeys(headlinesnews)):
        if x.text.strip() not in unwantedbs:
            speak(x.text.strip())

def getPerson(spoken_text):

    wordList = spoken_text.lower().split()

    for i in range(0, len(wordList)):
        if i + 3 <= len(wordList) - 1 and wordList[i].lower() == 'who' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]







def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        print("You said:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print("Error with the request; {0}".format(e))

    #changed var inside def from text to spoken_text
def extract_intent(spoken_text):
    #text = text.lower()
    spoken_text = spoken_text.lower()

    if "open" in spoken_text:
        intent = "open_app"
        app_name = extract_app_name(spoken_text)
        if app_name:
            return intent, app_name
        else:
            return None
    else:
        return None
    #open apps
##    if "open" in text:
##        app_name = extract_app_name(text)
##        return "open_app", app_name
##    #else
##    else:
##        #return chat for new function
##        return None
    
        

def extract_app_name(text):
    words = text.split()
    if "open" in words:
        index = words.index("open")
        if index + 1 < len(words):
            return words[index + 1]
    return None

def close_application(app_name):
    script = f'tell application "{app_name}" to quit'
    try:
        subprocess.run(['osascript', '-e', script])
        #print(f'{app_name} has been closed.')
    except Exception as e:
        print(f'Error closing {app_name}: {e}')
    return "closed"

def listen_for_wake_word():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for wake")
        audio = r.listen(source)

    try:
        intro = r.recognize_google(audio, language='en-us')
        print("Me  --> ", intro)
        return intro
    except:
        print("Me  -->  ERROR")
        return ""

def main():

    r = sr.Recognizer()


    while True:
        intro = listen_for_wake_word()

        if "Josh" in intro:
            os.system("afplay /Users/blakeweiss/Desktop/JOSH/aibeepmain.mp3")
            print("Taking input")
            spoken_text = ""
            with sr.Microphone() as source:
                audiotask = r.listen(source)

        

            try:
                spoken_text = r.recognize_google(audiotask, language='en-us')
                print("You said:", spoken_text)
            except:
                print("Me  -->  ERROR")

            #response = ""

            
            if spoken_text:
                #simple_llm.train(spoken_text)
                intent_info = extract_intent(spoken_text)
                if intent_info is not None:
                    intent, app_name = intent_info
                    response = handle_intent(intent, app_name)
             
                elif intent_info is None:                    
                    
                    if "are you there" in spoken_text.lower():
                        speak(f"Im here {MASTER}")

                    elif "what does Josh stand for" in spoken_text.lower():
                        speak("Josh stands for Just Ordinary Selfless Helper")
                    

                    #hide ip
                    elif "hide me" in spoken_text.lower():
                        os.system("bash /Users/blakeweiss/Desktop/JOSH/hidemeai.sh")
                        response = ("hiding your ip with tor")


                    #time
                    elif "time" in spoken_text.lower():
                        response = action_time()

                   

                    #who is
                    elif "who is" in spoken_text.lower():
                        person = getPerson(spoken_text)
                        wiki = wikipedia.summary(person, sentences = 2)
                        responsewhois = ("this is") + ' ' + wiki
                        response = responsewhois
                        print(responsewhois)

                    elif 'tell me a joke' in spoken_text.lower():
                        speak(pyjokes.get_joke())

                    #the news
                    elif any(i in spoken_text.lower() for i in ["tell me today's news", "tell me the news", "whats the news", "whats happening in the world"]):
                        speak("telling you the news")
                        getthenews()

                    elif any(i in spoken_text.lower() for i in ["how were you made", "what was your purpose", "what was the purpose of you being created"]):
                        howwereyoumade = ["I am a chatbot program with self thinking programing I have many skills avaiable for use.", "Im a ai chatbot program which should act as a assistent for whatever you may need."]
                        joshcreation = (random.choice(howwereyoumade))
                        speak(joshcreation + f"{MASTER}")

                    #search stuff
                    elif "search" in spoken_text.lower():
                        speak(f'What can I search for you {MASTER}?')
                        with sr.Microphone() as source:
                            speak("Just state what you want to search sir")
                            print("listening")
                            searchaudio = r.listen(source)
                            searchstuff="ERROR"
                        try:
                            searchstuff = r.recognize_google(searchaudio, language='en-us')
                            print("Me  --> ", searchaudio)
                        except:
                            print("Me  -->  ERROR")
                        response = webbrowser.open('https://www.google.com/search?q=' + searchstuff)


                    elif "movie" in spoken_text:
                        goodmovies = ["star wars", "jurrasic park", "clear and present danger", "war dogs", "wolf of wall street", "the big short", "trading places", "the gentlemen", "ferris bullers", "goodfellas", "lord of war", "borat", "marvel", "the hurt locker", "hustle", "forrest gump", "darkest hour", "coming to america", "warren miller movies", "the dictator"]
                        moviechoice = (random.choice(goodmovies))
                        response = ("A good movie you could watch is " + moviechoice + f" {MASTER}")

                    #skills avaiable
                    elif "what are your skills" in spoken_text:
                        skillsforuse = (
                        "-Hi, I am Josh. I can perform various tasks, including:\n"
                        "- Searching on Google\n"
                        "- Opening apps\n"
                        "- Telling you the date and time\n"
                        "- Running chat conversations\n"
                        "- Running 'hide me'\n"
                        "- Identifying people\n"
                        "- Searching for information\n"
                        "- Providing news updates\n"
                        "- Telling jokes"
                        )
                        response = (skillsforuse)
                        
                    elif "close" in spoken_text:
                        app_name = spoken_text.split("close ")[1]
                        close_application(app_name)
                    
                    if response:
                        speak(response)
                        print(response)
                    else:
                        #makes new chat run
                        #chatbot aspect
                        spaceholder = ("spaceholder1")
                        print(spaceholder)
            else:
                print("i dont understand")
                        
            #print("Response:", response)


if __name__ == "__main__":
    main()
