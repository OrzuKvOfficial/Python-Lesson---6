from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Chatbotni yaratish
chatbot = ChatBot('SimpleBot')

# Trenerni sozlash
trainer = ChatterBotCorpusTrainer(chatbot)

# Chatbotni inglizcha korpus bilan o'qitish
trainer.train("chatterbot.corpus.english")

# Chatbot bilan muloqot qilish
while True:
    try:
        user_input = input("Siz: ")
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

    except (KeyboardInterrupt, EOFError, SystemExit):
        break
