import os
import queue
import sounddevice as sd
import vosk
import sys
import json
import random
import torch
from art import tprint

from tts import *
from functions import *
from ai_sort import *

from LLM2 import generate_answer

samplerate = 16000  
device = None       
channels = 1
chunk = 1024  

tts=TTS()

model_path = "vosk-model-ru-0.10"
if not os.path.exists(model_path):
    print(f"Модель не найдена по пути '{model_path}'")
    sys.exit(1)

model = vosk.Model(model_path)

q = queue.Queue()

netschool_login = os.getenv("NETSCHOOL_LOGIN")
netschool_password = os.getenv("NETSCHOOL_PASSWORD")



def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

rec = vosk.KaldiRecognizer(model, samplerate)

thanks = ["urwelcome", "urwelcome2", "urwelcome3"]
done = ['done1', 'done2', 'done3']
callbacks = ['callback1', 'callback2', 'callback3']
hello = ['привет', 'приветствую', 'здарова','здаров','даров','прив','здравствуй','здравствуйте']
request_count = 0
flag_browser = False
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                       channels=channels, callback=callback):
    tprint("WELCOME VITALY\n")
    voice_fast_callback('ready', chunk)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = json.loads(result).get("text", "")
            
            print(text)
            
            if "благодарю" in text or "спасибо" in text.lower():
                voice_fast_callback(random.choice(thanks), chunk)   
                
            if flag_browser:
                if text.strip() == "":
                    continue
                search_web(text)
                flag_browser = False    
                voice_fast_callback(random.choice(done), chunk)
            if text == 'кен' or text == 'кенн' or text == 'кеннеди':
              voice_fast_callback(random.choice(callbacks), chunk)
            if 'кен' in text and not flag_browser:
              text = text.replace('кен', '')
              ans = query_classify(text)
              
              for i in hello: 
                if i in text:
                  voice_fast_callback('hello-night', chunk)
              
              if ans == 'command':
                # pass
                ai_sort = final_query_handler(text)
                
                if ai_sort == 'браузер':
                  voice_fast_callback('browser-run', chunk)
                  voice_fast_callback('browser-url', chunk)
                  flag_browser = True
                  
                  continue
                elif ai_sort == 'погода':
                  weather = get_weather('Южно-Сахалинск')
                  tts.text2speech(weather)
                elif ai_sort == 'время':
                  tts.text2speech(f'сейчас {get_time()}, сэр')
                elif ai_sort == 'стим':
                  pass
                elif ai_sort == 'музыка':
                  pass
                elif ai_sort == 'расписание':
                  get_schedule_file()
                  directory = 'schedules'
                  filename = None

                  for file in os.listdir(directory):
                      if file.startswith("!!Расписание"):
                          filename = file
                          break
                  if filename:
                    tts.text2speech("Файл расписания найден. вы хотите открыть его?")
                  else:
                    tts.text2speech("Файл расписания не найден. Возможно его ещё не сделали")
                  
              elif ans == 'dialog':
                tts.text2speech(generate_answer(text))
              else:
                pass
              
        else:
            partial_result = rec.PartialResult()
            