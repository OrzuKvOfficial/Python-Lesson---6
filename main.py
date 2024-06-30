def main():
    while True:
        try:
            # Foydalanuvchidan son kiritishni so'raymiz
            num = int(input("Son kiriting: "))
            
            # Agar foydalanuvchi 0 kiritgan bo'lsa, noldan bo'lish xatosini chaqiramiz
            result = 10 / num
            print(f"Natija: {result}")
        
        except ValueError:
            # Agar foydalanuvchi noto'g'ri ma'lumot kiritsa, bu xabar chiqadi
            print("Xato: Iltimos, son kiriting.")
        
        except ZeroDivisionError:
            # Agar foydalanuvchi 0 kiritsa, bu xabar chiqadi
            print("Xato: Nolga bo'linish mumkin emas.")
        
        except Exception as e:
            # Boshqa turdagi xatolar uchun umumiy xabar
            print(f"Xato: {e}")
        
        # Davom etishni xohlaysizmi?
        continue_prompt = input("Davom etishni xohlaysizmi? (ha/yo'q): ")
        if continue_prompt.lower() != 'ha':
            break

if __name__ == "__main__":
    main()
