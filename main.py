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
