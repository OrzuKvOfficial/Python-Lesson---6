import os
import time

def check_usb_drives():
    drives_before = set(os.listdir('/media'))
    
    while True:
        time.sleep(2)  # Har 2 soniyada tekshiradi
        drives_after = set(os.listdir('/media'))
        new_drives = drives_after - drives_before
        
        if new_drives:
            print(f"Yangi USB qurilmasi ulandi: {new_drives}")
            return list(new_drives)
        
        drives_before = drives_after

if __name__ == "__main__":
    print("USB qurilmasini ulashni kutyapman...")
    new_drives = check_usb_drives()
    print(f"Ulangan yangi USB qurilmasi: {new_drives}")
