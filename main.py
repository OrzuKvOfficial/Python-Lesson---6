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
import cv2
import os

# Yuzni aniqlash uchun oldindan tayyorlangan ma'lumotlar to'plami
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuzni aniqlash va rasmga olish funksiyasi
def capture_face(employee_id):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Yuzni aniqlash")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Yuzni aniqlash", frame)

        if len(faces) > 0:
            img_name = f"employee_{employee_id}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saqlandi!")
            break

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC tugmasini bosish
            print("Chiqildi!")
            break

    cam.release()
    cv2.destroyAllWindows()

# Ishchi ID raqamini kiriting
employee_id = input("Iltimos, ishchi ID raqamini kiriting: ")
capture_face(employee_id)
import ast
import subprocess

def analyze_code(code):
    try:
        # AST (Abstract Syntax Tree) yordamida tahlil qilish
        tree = ast.parse(code)
        print("AST analysis successful!")
    except SyntaxError as e:
        print(f"Syntax error: {e}")

    # flake8 yordamida kodni tahlil qilish
    result = subprocess.run(['flake8', '--stdin-display-name', 'stdin', '-'], input=code, text=True, capture_output=True)
    if result.stdout:
        print("Flake8 Analysis:")
        print(result.stdout)
    else:
        print("No issues found by Flake8.")

    # pylint yordamida kodni tahlil qilish
    result = subprocess.run(['pylint', '--from-stdin'], input=code, text=True, capture_output=True)
    if result.stdout:
        print("Pylint Analysis:")
        print(result.stdout)
    else:
        print("No issues found by Pylint.")
