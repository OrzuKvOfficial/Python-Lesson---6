# List yaratish
my_list = [1, 2, 3, 4, 5]

# List elementlariga murojaat qilish
print(my_list[0])  # Birinchi element
print(my_list[-1])  # So'nggi element

# Listga yangi element qo'shish
my_list.append(6)
print(my_list)

# Listdan elementni olib tashlash
my_list.remove(3)
print(my_list)

# Listni birlashtirish
another_list = [7, 8, 9]
combined_list = my_list + another_list
print(combined_list)

# Listni tartiblash
my_list.sort()
print(my_list)
my_list.sort(reverse=True)
print(my_list)

# Listni aylantirish (iteratsiya)
for element in my_list:
    print(element)

# List comprehension
squares = [x**2 for x in range(10)]
print(squares)

# List uzunligini aniqlash
length_of_list = len(my_list)
print(length_of_list)

# List elementlarini tekshirish
if 3 in my_list:
    print("3 ro'yxatda bor")
else:
    print("3 ro'yxatda yo'q")

# Listdan kesmalar olish
sub_list = my_list[1:4]
print(sub_list)
