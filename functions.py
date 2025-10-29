import datetime
import requests
import webbrowser
from translate import Translator

def get_weather(city):
    api_key = '9c5258c859de394817b5b82cd64a0455' # получите ключ API на сайте OpenWeatherMap
    url = 'https://api.openweathermap.org/data/2.5/weather?q='+city+'&units=metric&lang=ru&appid=79d1ca96933b0328e1c7e3e7a26cb347'
    
    try:
        response = requests.get(url)
        response.raise_for_status() # сгенерирует исключение для кода отличного от 200
        
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        return f'Погода в городе {city}: {weather_description}, Температура: {temperature}°C, Влажность: {humidity}%'
    
    except requests.exceptions.HTTPError as err:
        # обработка ошибки
        return f'Произошла ошибка: {err}'

#Открытие гугла
def search_web(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)

def translate_text_to_rus(text, from_lang='en', to_lang='ru'):
    # Создаем объект Translator, указывая исходный язык и язык перевода
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    try:
        # Пытаемся перевести текст
        translated_text = translator.translate(text)
        return translated_text  # Возвращаем переведенный текст
    except Exception as e:
        # Если возникает ошибка, возвращаем сообщение об ошибке
        return f"Error: {e}"
    
def translate_text_to_eng(text, from_lang='ru', to_lang='en'):
    # Создаем объект Translator, указывая исходный язык и язык перевода
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    try:
        # Пытаемся перевести текст
        translated_text = translator.translate(text)
        return translated_text  # Возвращаем переведенный текст
    except Exception as e:
        # Если возникает ошибка, возвращаем сообщение об ошибке
        return f"Error: {e}"

def get_time():
    now = datetime.datetime.now()
    return now.strftime("%H часов %M минут")
