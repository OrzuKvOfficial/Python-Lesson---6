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
