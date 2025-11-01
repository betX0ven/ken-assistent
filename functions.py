import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
import os
import datetime
import requests
import webbrowser
from translate import Translator
import subprocess

def get_weather(city):
    api_key = '9c5258c859de394817b5b82cd64a0455'
    url = 'https://api.openweathermap.org/data/2.5/weather?q='+city+'&units=metric&lang=ru&appid=79d1ca96933b0328e1c7e3e7a26cb347'
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        
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


def login_netschool(browser, login, password):
  browser.find_element(By.XPATH, '/html/body/div[1]/div[1]/div[4]/a[2]').click()
  WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, 'login'))).send_keys(login)
  WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, 'password'))).send_keys(password)
  browser.find_element(By.CLASS_NAME, 'plain-button').click()
  WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'primary-button'))).click()
  try:
    print(1)
    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[3]/div/div/ns-modal/div/div[3]/div/div/button"))).click()
    print(2)
  except Exception as e:
    pass

download_dir = os.path.abspath("schedules")
if not os.path.exists(download_dir):
  os.makedirs(download_dir)
  
options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_dir,  
    "download.prompt_for_download": False,     
    "download.directory_upgrade": True,
    "safebrowsing.enabled": False,
    "profile.default_content_settings.popups": 0, 
    "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,              
}
options.add_experimental_option("prefs", prefs)
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--enable-automation")

def get_schedule_file():
  with webdriver.Chrome(options=options) as browser:
    browser.get('https://netcity.admsakhalin.ru:11111/')
    try:
      login_netschool(browser, netschool_login, netschool_password)
      WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="2"]/div[3]/div/a'))).click()
    #   WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'remembered-users__item'))).click()
      schedule = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '//a[starts-with(text(),"!!Расписание")]')))
      print(schedule.text)
      schedule.click()
    except Exception as e:
      print(f'Произошла ошибка: {e}')

netschool_login = os.getenv("NETSCHOOL_LOGIN")
netschool_password = os.getenv("NETSCHOOL_PASSWORD")


# get_schedule_file()

def open_program(program_name):
    if program_name == 'steam':
        os.startfile(r"programs\Steam.lnk")
    elif program_name == 'telegram':
        webbrowser.open("tg:")
    elif program_name == 'spotify':
        os.startfile(r"programs\Spotify.lnk")
    elif program_name == 'vk':
        webbrowser.open("vk.com")
    elif program_name == 'whatsapp':
        webbrowser.open("whatsapp:")
    elif program_name == 'netcity':
        webbrowser.open("https://netcity.admsakhalin.ru:11111/")
    elif program_name == 'wildberries':
        webbrowser.open("https://www.wildberries.ru/")
    elif program_name == 'youtube':
        webbrowser.open("https://www.youtube.com/")
    else:
        return f"Программа {program_name} не найдена"
        
def spotify():
    pass
