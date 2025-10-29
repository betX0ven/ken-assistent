import os
import queue
import sounddevice as sd
import vosk
import sys
import json
import random
import torch

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

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

rec = vosk.KaldiRecognizer(model, samplerate)

thanks = ["urwelcome", "urwelcome2", "urwelcome3"]
hello = ['привет', 'приветствую', 'здарова','здаров','даров','прив','здравствуй','здравствуйте']
request_count = 0

with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                       channels=channels, callback=callback):
    print("Начинаю распознавание. Говорите...\n")
    voice_fast_callback('ready', chunk)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = json.loads(result).get("text", "")
            
            print(text)
            
            if "благодарю" in text or "спасибо" in text.lower():
                voice_fast_callback(random.choice(thanks), chunk)   
                
            if 'кен' in text:
              text = text.replace('кен', '')
              ans = query_classify(text)
              
              for i in hello: 
                if i in text:
                  voice_fast_callback('hello-night', chunk)
              
              if ans == 'command':
                pass
              elif ans == 'dialog':
                tts.text2speech(generate_answer(text))
              else:
                pass
              
        else:
            partial_result = rec.PartialResult()
            