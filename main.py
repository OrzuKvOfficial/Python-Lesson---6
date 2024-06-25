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
def main(file_path):
    code = read_code(file_path)
    analyze_code(code)

if __name__ == "__main__":
    file_path = 'your_code.py'  # Bu yerda tahlil qilinadigan fayl nomini kiriting
    main(file_path)
import cv2
import face_recognition

# Video olish uchun kamera ochish
video_capture = cv2.VideoCapture(0)

while True:
    # Videodan bir frame o'qish
    ret, frame = video_capture.read()

    # Frame'ni RGB formatiga o'zgartirish (OpenCV BGR formatidan foydalanadi)
    rgb_frame = frame[:, :, ::-1]

    # Frame'dagi barcha yuzlarni topish
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        # Yuzning atrofini chizish
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Oynada natijani ko'rsatish
    cv2.imshow('Video', frame)

    # 'q' tugmasini bosish orqali chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Videoni yakunlash
video_capture.release()
cv2.destroyAllWindows()
import pygame
import random

# Pygame ni boshlash
pygame.init()

# O'yin oynasi o'lchamlari
screen_width = 800
screen_height = 600

# Ranglar
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Ekranni yaratish
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Pyda Qanqadur Soda O\'yini')

# FPS
clock = pygame.time.Clock()
snake_block = 10
snake_speed = 15

font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)


def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, black)
    screen.blit(value, [0, 0])


def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, black, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [screen_width / 6, screen_height / 3])


def gameLoop():
    game_over = False
    game_close = False

    x1 = screen_width / 2
    y1 = screen_height / 2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close == True:
            screen.fill(white)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            your_score(Length_of_snake - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= screen_width or x1 < 0 or y1 >= screen_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        screen.fill(white)
        pygame.draw.rect(screen, red, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        your_score(Length_of_snake - 1)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()


gameLoop()
# Ustun nomlarini o'zgartirish
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Yangi ustun qo'shish
df['new_column'] = df['existing_column'] * 2

# Ustunni o'chirish
df.drop(columns=['column_to_drop'], inplace=True)

# Ustun nomlarini o'zgartirish
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Yangi ustun qo'shish
df['new_column'] = df['existing_column'] * 2

# Ustunni o'chirish
df.drop(columns=['column_to_drop'], inplace=True)

import pandas as pd
import numpy as np

# 1. Ma'lumotlarni yuklash
df = pd.read_csv('employees.csv')

# 2. Ma'lumotlarni tahlil qilish
print("Dastlabki 5 qator:")
print(df.head())

print("\nMa'lumotlar to'plami haqida:")
print(df.info())

print("\nStatistik ma'lumotlar:")
print(df.describe())

# 3. Ustunlar va qatorlarni taxrirlash
# Ustun nomlarini o'zgartirish
df.rename(columns={'Name': 'Full Name', 'Salary': 'Annual Salary'}, inplace=True)

# Yangi ustun qo'shish (masalan, yillik bonus 10% maosh asosida)
df['Bonus'] = df['Annual Salary'] * 0.10

# Shartga ko'ra qatorlarni filtr qilish (masalan, yoshi 30 dan katta bo'lgan xodimlar)
df_above_30 = df[df['Age'] > 30]

# Jins bo'yicha guruhlash va har bir guruh uchun o'rtacha maoshni hisoblash
average_salary_by_gender = df.groupby('Gender')['Annual Salary'].mean()

print("\nJins bo'yicha o'rtacha maosh:")
print(average_salary_by_gender)
# Bo'sh qiymatlar bilan ishlash
# Bo'sh qiymatlarni to'ldirish (masalan, bo'sh yoshi uchun o'rtacha yoshni qo'yish)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Maosh bo'sh bo'lsa, uni 0 ga to'ldirish
df['Annual Salary'].fillna(0, inplace=True)

# Bo'sh qiymatlarni o'chirish (masalan, bo'sh ismli qatorlarni o'chirish)
df.dropna(subset=['Full Name'], inplace=True)

# Qatorlarni aralash va qayta indekslash
df = df.sample(frac=1).reset_index(drop=True)

# 4. Yana qo'shimcha taxrirlash
# Maoshni valyuta formatida ifodalash
df['Annual Salary'] = df['Annual Salary'].apply(lambda x: "${:,.2f}".format(x))

# Yoshi katta va kichik 5 ta xodimni ko'rish
oldest_5 = df.nlargest(5, 'Age')
youngest_5 = df.nsmallest(5, 'Age')

print("\nYoshi katta 5 xodim:")
print(oldest_5)

print("\nYoshi kichik 5 xodim:")
print(youngest_5)

# 5. Ma'lumotlarni saqlash
df.to_csv('edited_employees.csv', index=False)

import pandas as pd
from datetime import datetime

# Dastlabki ma'lumotlar faylini yaratamiz yoki mavjud bo'lsa yuklaymiz
try:
    health_data = pd.read_csv('health_data.csv')
except FileNotFoundError:
    columns = ['Date', 'Water (L)', 'Sleep (hours)', 'Exercise (minutes)', 'Notes']
    health_data = pd.DataFrame(columns=columns)

# Foydalanuvchi ma'lumotlarini kiritish
date = datetime.now().strftime('%Y-%m-%d')
water = float(input("Bugun qancha suv ichdingiz? (L): "))
sleep = float(input("Kecha qancha soat uxladingiz? (hours): "))
exercise = float(input("Bugun qancha daqiqa jismoniy mashqlar qildingiz? (minutes): "))
notes = input("Bugungi sog'lig'ingiz haqida izohlar: ")

# Yangi ma'lumotlarni DataFrame ga qo'shish
new_data = pd.DataFrame([[date, water, sleep, exercise, notes]], columns=health_data.columns)
health_data = pd.concat([health_data, new_data], ignore_index=True)

# Ma'lumotlarni CSV fayliga saqlash
health_data.to_csv('health_data.csv', index=False)

print("Ma'lumotlaringiz saqlandi. Sog'lig'ingizni kuzatishda davom eting!")
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

class PydaAqili:
    def __init__(self):
        self.name = "Pyda Aqili"
        self.greetings = ["hello", "hi", "hey", "greetings", "sup", "what's up"]
        self.responses = {
            "how are you": "I'm an AI, so I don't have feelings, but thanks for asking!",
            "what is your name": "I am Pyda Aqili, your friendly AI assistant.",
            "bye": "Goodbye! Have a nice day!"
        }
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens
    
    def get_response(self, user_input):
        processed_input = self.preprocess_text(user_input)
        for word in processed_input:
            if word in self.greetings:
                return "Hello! How can I assist you today?"
            elif word in self.responses:
                return self.responses[word]
        return "I'm sorry, I don't understand that."

# Example interaction
bot = PydaAqili()
print(bot.get_response("Hello"))
print(bot.get_response("What is your name?"))
print(bot.get_response("How are you?"))
print(bot.get_response("Bye"))
def divide(a, b):
    import pdb; pdb.set_trace()  # Bu yerda ushlab tekshirishni boshlaydi
    return a / b

result = divide(10, 0)
print(result)
import matplotlib.pyplot as plt
import numpy as np

def pyda_budunlugu_boyut_hesapla(uzunluk, genislik):
    # Kumaş miktarını hesaplama (iki parça için)
    toplam_alan = 2 * (uzunluk * genislik)
    return toplam_alan

def pyda_budunlugu_sablon_ciz(uzunluk, genislik):
    # Şablon çizme
    fig, ax = plt.subplots()
    oval = plt.Circle((genislik / 2, uzunluk / 2), radius=min(uzunluk, genislik) / 2, edgecolor='r', facecolor='none')
    ax.add_patch(oval)
    plt.xlim(0, genislik)
    plt.ylim(0, uzunluk)
    ax.set_aspect('equal', adjustable='box')
    plt.title('Pyda Budunluğu Şablonu')
    plt.show()

# Pyda budunluğu boyutları (cm cinsinden)
uzunluk = 80  # uzunluk
genislik = 50  # genişlik

# Kumaş miktarını hesaplama
toplam_kumas_alani = pyda_budunlugu_boyut_hesapla(uzunluk, genislik)
print(f"Toplam kumaş alanı: {toplam_kumas_alani} cm^2")

# Şablonu çizme
pyda_budunlugu_sablon_ciz(uzunluk, genislik)
my_list = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}, {'name': 'Charlie', 'age': 20}]
sorted_list = sorted(my_list, key=lambda x: x['age'])
print(sorted_list)
import time

def measure_sort_time(sort_function, data):
    start_time = time.time()
    sort_function(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Misol uchun ro'yxat yaratamiz
import random
data_size = 1000
random_data = [random.randint(0, 1000) for _ in range(data_size)]

# Saralash algoritmlari uchun vaqtni o'lchaymiz
bubble_sort_time = measure_sort_time(bubble_sort, random_data.copy())
insertion_sort_time = measure_sort_time(insertion_sort, random_data.copy())
quick_sort_time = measure_sort_time(quick_sort, random_data.copy())

print(f"Bubble sort vaqti: {bubble_sort_time:.6f} soniya")
print(f"Insertion sort vaqti: {insertion_sort_time:.6f} soniya")
print(f"Quick sort vaqti: {quick_sort_time:.6f} soniya")
import os

def pyda_katta_project_info(directory):
    project_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pyda") or file.endswith(".katta"):
                file_path = os.path.join(root, file)
                project_files.append(file_path)

    return project_files

# Istalgan direktoriyani kiriting
directory = "/path/to/your/project"
project_files = pyda_katta_project_info(directory)

if project_files:
    print("Pyda Katta Project fayllari:")
    for file in project_files:
        print(file)
else:
    print("Hech qanday Pyda Katta Project fayli topilmadi.")
from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text('Salom, men botman!')

updater = Updater('YOUR_TOKEN', use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))

updater.start_polling()
updater.idle()
from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text('Salom, men botman!')

def main():
    # Telegram bot tokenini kiriting
    updater = Updater('YOUR_TOKEN', use_context=True)
    dp = updater.dispatcher

    # /start komandasini ishlashini o'rnatish
    dp.add_handler(CommandHandler('start', start))

    # Botni ishga tushirish
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
def help_command(update, context):
    update.message.reply_text('Bu bot sizga yordam beradi. Komandalar:\n/start - Boshlash\n/help - Yordam')

def main():
    updater = Updater('YOUR_TOKEN', use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help_command))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
from telegram.ext import MessageHandler, Filters

def echo(update, context):
    update.message.reply_text(update.message.text)

def main():
    updater = Updater('YOUR_TOKEN', use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
import openai
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# OpenAI API kalitini kiriting
openai.api_key = 'YOUR_OPENAI_API_KEY'

def gpt3_response(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def echo(update, context):
    user_message = update.message.text
    bot_response = gpt3_response(user_message)
    update.message.reply_text(bot_response)

def main():
    updater = Updater('YOUR_TELEGRAM_BOT_TOKEN', use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
class ToDoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("To-Do List Application")

        self.tasks = []

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.listbox = tk.Listbox(self.frame, height=15, width=50)
        self.listbox.pack(side=tk.LEFT)

        self.scrollbar = tk.Scrollbar(self.frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)

        self.entry = tk.Entry(root, width=50)
        self.entry.pack()

        self.add_button = tk.Button(root, text="Add Task", width=48, command=self.add_task)
        self.add_button.pack()

        self.delete_button = tk.Button(root, text="Delete Task", width=48, command=self.delete_task)
        self.delete_button.pack()

    def add_task(self):
        task = self.entry.get()
        if task != "":
            self.tasks.append(task)
            self.update_listbox()
            self.entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Warning", "You must enter a task.")

    def delete_task(self):
        try:
            task_index = self.listbox.curselection()[0]
            self.tasks.pop(task_index)
            self.update_listbox()
        except:
            messagebox.showwarning("Warning", "You must select a task to delete.")

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for task in self.tasks:
            self.listbox.insert(tk.END, task)

if __name__ == "__main__":
    root = tk.Tk()
    app = ToDoApp(root)
    root.mainloop()
# Sorting a list in ascending order
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
my_list.sort()
print(my_list)  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

# Sorting a list in descending order
my_list.sort(reverse=True)
print(my_list)  # Output: [9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1]
numbers = [3, 1, 4, 2]
numbers.sort()
print(numbers)  # [1, 2, 3, 4]

# Yoki
numbers = [3, 1, 4, 2]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # [1, 2, 3, 4]
# Bo'sh list yaratish
my_list = []

# Elementlar bilan list yaratish
my_list = [1, 2, 3, 4, 5]

# Oxiriga element qo'shish
my_list.append(6)

# Belgilangan indeksga element qo'shish
my_list.insert(2, 'new_element')  # 2-indeksga yangi element qo'shildi

# Belgilangan indeksdagi elementni o'chirish
del my_list[3]

# Berilgan qiymatni o'chirish
my_list.remove('new_element')

# Belgilangan indeksdagi elementni o'zgartirish
my_list[1] = 'changed_element'

# Listni bo'ylab yurish
for element in my_list:
    print(element)

# List uzunligini aniqlash
length_of_list = len(my_list)
print(f"List uzunligi: {length_of_list}")

# Listni bo'lish (slice)
sub_list = my_list[1:4]  # 1-indeksdan 4-indeksgacha bo'lgan elementlarni oladi (4-indeks o'z ichiga olmagan holda)
print(f"Sub-list: {sub_list}")
