from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import pymorphy3
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib



morph = pymorphy3.MorphAnalyzer()

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

all_answers = ['погода', 'время', 'стим', 'браузер', 'музыка']
banword = ["как, он, мы, его, вы, вам, вас, ее, что, который, их, все, они, я, весь, мне, меня, таким, для, на, по, со, из, от, до, без, над, под, за, при, после, во, же, то, бы, всего, итого, даже, да, нет, ой, ого, эх, браво, здравствуйте, спасибо, извините, пожалуйста".replace(",","").split()][0]

tags = [i for i in all_answers]
text_embeddings = model.encode(tags, convert_to_tensor=True)

def semantic_search(query, embeddings, top_n=3):
    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    top_results = np.argsort(similarities.cpu().numpy())[-top_n:][::-1]
    
    if float(similarities[top_results[0]]) > 0.53:
        return all_answers[top_results[0]]
    else:
        return "for_ai"
    
def start_ai(query):
    response = semantic_search(query, text_embeddings)
    return response

def preparing_query(query):
    total_query = ''
    for word in query.split():
        word = morph.parse(word)[0]
        total_query+=word.normal_form+" "
    return total_query

def final_query_handler(query):
    print('Запрос: ' + query)
    return start_ai(preparing_query(query))


training_data = [
    # Команды
    ("создай 3d модель реактора", "command"),
    ("включи все системы", "command"),
    ("просканируй территорию", "command"),
    ("рассчитай траекторию полёта", "command"),
    ("покажи голограмму", "command"),
    ("активируй протокол", "command"),
    ("запусти симуляцию", "command"),
    ("выключи питание", "command"),
    ("построй график", "command"),
    ("найди уязвимости", "command"),
    
    # Диалоги
    ("как думаешь это сработает", "dialog"),
    ("почему не работает система", "dialog"),
    ("что если попробовать другой способ", "dialog"),
    ("интересно почему он так поступил", "dialog"),
    ("может быть нам стоит передумать", "dialog"),
    ("как тебе такая идея", "dialog"),
    ("объясни принцип работы", "dialog"),
    ("зачем нам это нужно", "dialog"),
    ("что ты об этом думаешь", "dialog"),
    ("как улучшить конструкцию", "dialog"),
    ("расскажи про это", "dialog"),
    
    # Сложные случаи
    ("проанализируй и скажи мнение", "command"), # команда + диалог = команда
    ("расскажи как это работает", "dialog"), # приоритет командам
]

class QueryClassify():
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  
                max_features=1000
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
    def preparing(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text) 
        return text
    
    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)
    
    def predict(self, text):
        prepared_query = self.preparing(text)
        
        prediction = self.pipeline.predict([prepared_query])[0]
        probability = np.max(self.pipeline.predict_proba([prepared_query]))
        
        return prediction, probability
    
    def save_model(self, path):
        joblib.dump(self.pipeline, path)
    
    def load_model(self, path):
        pass

def create_and_train_classifier():
    # Разделяем данные на тексты и метки
    texts, labels = zip(*training_data)
    
    # Создаем и обучаем классификатор
    classifier = QueryClassify()
    classifier.train(texts, labels)
    
    return classifier


def test_classifier(classifier):
    test_phrases = [
        "создай новую броню для меня",
        "как думаешь какой металл лучше",
        "включи систему невидимости сейчас же",
        "почему не работает реактор",
        "может стоит попробовать вибраниум",
        "проанализируй эту схему и дай оценку",
        "что если мы всё делаем неправильно",
        "запусти протокол очистки памяти",
        "интересно сработает ли этот алгоритм",
        "построй график энергопотребления"
    ]

    for phrase in test_phrases:
        intent, confidence = classifier.predict(phrase)
        status = "✅" if confidence > 0.7 else "⚠️"


def query_classify(query):
    classifier = create_and_train_classifier()
    test_classifier(classifier)
    if os.path.exists("intent_classifier.pkl"):
        classifier.load_model("intent_classifier.pkl")
    else:
        classifier = create_and_train_classifier()
        classifier.save_model("intent_classifier.pkl")
        
    intent, confidence = classifier.predict(query)
    if intent == "unclear":
        print("Хз")
        return 'null'
    elif intent == "command":
        print(f"Команда, {confidence:.2f})")
        return 'command'
    else:  # dialog
        print(f"Диалог, {confidence:.2f})")
        return 'dialog'
