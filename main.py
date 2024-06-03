import speech_recognition as sr

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Буйруқни айтиб кўринг...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio, language="uz-UZ")
        print("Сизнинг буйруғингиз: ", command)
        return command
    except sr.UnknownValueError:
        print("Буйруқ тушунарсиз.")
    except sr.RequestError:
        print("Сервисга уланишда хатолик.")
    return ""

# Микроконтроллерга уланып, моторларни бошқариш учун код ёзиш мумкин.
import matplotlib.pyplot as plt

# Histogram chizish
df['column_name'].hist()
plt.title('Histogram of Column Name')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Chiziqli grafik chizish
df.plot(x='x_column', y='y_column', kind='line')
plt.title('Line Plot of X vs Y')
plt.xlabel('X Column')
plt.ylabel('Y Column')
plt.show()
import socket

def start_server(host='127.0.0.1', port=65432):
    # Socketni yaratish
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))  # Serverni IP va portga bog'lash
        s.listen()  # Kiruvchi ulanishlarni tinglash
        print(f"Server {host}:{port} manzilida ishlamoqda...")

        conn, addr = s.accept()  # Ulanish qabul qilinadi
        with conn:
            print(f"Ulangan: {addr}")
            while True:
                data = conn.recv(1024)  # Ma'lumotlarni qabul qilish (1024 baytgacha)
                if not data:
                    break  # Agar ma'lumot bo'lmasa, loopdan chiqish
                print(f"Olingan ma'lumot: {data.decode()}")  # Ma'lumotni dekodlash va chop etish

if __name__ == "__main__":
    start_server()
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Matnni tayyorlash funksiyasi
def prepare_text(text):
    # Kichik harflar
    text = text.lower()
    # Maxsus belgilarni olib tashlash
    text = re.sub(r'\W', ' ', text)
    # Bir belgili so'zlarni olib tashlash
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Bir nechta bo'sh joylarni olib tashlash
    text = re.sub(r'\s+', ' ', text)
    return text

# Namuna matnlar
documents = [
    "Pythonda dasturlash juda qiziqarli va foydali.",
    "Matinlarni tahlil qilish muhim va qiziqarli mavzu.",
    "AI va Machine Learning texnologiyalari bugungi kunda juda dolzarbdir.",
    "OpenAI tomonidan ishlab chiqilgan GPT-3 modeli matnlarni tahlil qilishda juda yaxshi natijalarga erishmoqda."
]

# Matnlarni tayyorlash
documents = [prepare_text(doc) for doc in documents]

# TF-IDF vektorlashtirish
vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
tfidf_matrix = vectorizer.fit_transform(documents)

# LDA modeli yordamida asosiy mavzularni aniqlash
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tfidf_matrix)

# Har bir mavzu uchun asosiy so'zlarni chiqarish
def print_topics(model, vectorizer, top_n=10):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"Mavzu {idx+1}:")
        print(" ".join([words[i] for i in topic.argsort()[:-top_n - 1:-1]]))

print_topics(lda, vectorizer)
