import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def ask_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Yoki mavjud boshqa engine nomi
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    return answer

if __name__ == "__main__":
    print("Savolingizni kiriting (chiqish uchun 'exit' deb yozing):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        answer = ask_gpt(user_input)
        print("Bot:", answer)
