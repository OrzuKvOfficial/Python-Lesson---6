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
# 1 dan 10 gacha bo'lgan sonlar ro'yxatini yaratish
numbers = [i for i in range(1, 11)]
print(numbers)

# Har bir elementning kvadratini olish
squares = [x**2 for x in numbers]
print(squares)

# Faqat juft sonlarni olish
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)
# 3x3 matritsa yaratish
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Matritsani chiroyli chiqarish
for row in matrix:
    print(row)

# Matritsadagi barcha elementlarning yig'indisini hisoblash
total = sum(sum(row) for row in matrix)
print(f"Yig'indi: {total}")

# Matritsani tekis (flatten) qilish
flattened_matrix = [item for row in matrix for item in row]
print(flattened_matrix)
