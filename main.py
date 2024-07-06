import tkinter as tk
from time import strftime

# Oynani yaratish
root = tk.Tk()
root.title("Soat")

# Soat funksiyasini yaratish
def time():
    string = strftime('%H:%M:%S %p')
    label.config(text=string)
    label.after(1000, time)

# Labelni yaratish
label = tk.Label(root, font=('calibri', 40, 'bold'), background='purple', foreground='white')
label.pack(anchor='center')

# Soat funksiyasini chaqirish
time()

# Oynani ishga tushirish
root.mainloop()
