import sounddevice as sd
import torch
import time
import pyaudio  
import wave  
import re
from num2words import num2words

# константы голосов, поддерживаемых в silero
SPEAKER_AIDAR   = "aidar"
SPEAKER_BAYA    = "baya"
SPEAKER_KSENIYA = "kseniya"
SPEAKER_XENIA   = "xenia"
SPEAKER_RANDOM  = "random"

# константы девайсов для работы torch
DEVICE_CPU    = "cpu"
DEVICE_CUDA   = "cuda" 
DEVICE_VULKAN = "vulkan"
DEVICE_OPENGL = "opengl"
DEVICE_OPENCL = "opencl"

class TTS:
    def __init__(
            self, speaker: str = SPEAKER_AIDAR, 
            device: str        = DEVICE_CPU, 
            samplerate: int    = 48_000
        ):
        
        # подгружаем модель 
        self.__MODEL__, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="ru_v3"
        )
        self.__MODEL__.to(torch.device(device))

        self.__SPEAKER__ = speaker
        self.__SAMPLERATE__ = samplerate
    
    def text2speech(self, text: str):
        text = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='ru'), text)
        audio = self.__MODEL__.apply_tts(
            text=text,               
            speaker=self.__SPEAKER__,
            sample_rate=self.__SAMPLERATE__, 
            put_accent=True,
            put_yo=True
        )

        sd.play(audio, samplerate=self.__SAMPLERATE__)
        time.sleep((len(audio)/self.__SAMPLERATE__))
        sd.stop()
        
        
chunk = 1024  
def voice_fast_callback(ans_text, chunk):
  try:
    #open a wav format music  
    f = wave.open(f"answers/{ans_text}.wav","rb")  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  
      
    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
      
    #stop stream  
    stream.stop_stream()  
    stream.close()  
      
    #close PyAudio  
    p.terminate()  
  except Exception:
    #open a wav format music  
    f = wave.open(f"answers/error.wav","rb")  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  
      
    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
      
    #stop stream  
    stream.stop_stream()  
    stream.close()  
      
    #close PyAudio  
    p.terminate()  
  