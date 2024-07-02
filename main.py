from textblob import TextBlob

def check_text(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return corrected_text

# Misol uchun, foydalanuvchidan matn olish va uni tekshirish
user_input = input("Matin kiriting: ")
corrected_text = check_text(user_input)
print(f"Tuzatilgan matn: {corrected_text}")
from textblob import TextBlob

def check_text(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return corrected_text

# Misol uchun, foydalanuvchidan matn olish va uni tekshirish
user_input = input("Matin kiriting: ")
corrected_text = check_text(user_input)
print(f"Tuzatilgan matn: {corrected_text}")
