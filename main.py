import zipfile
import os

def zip_files(directory, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))

# Misol uchun, "my_folder" papkasidagi barcha fayllarni "output.zip" ga siqish
directory_to_zip = 'my_folder'
output_zip_file = 'output.zip'
zip_files(directory_to_zip, output_zip_file)

print(f"{output_zip_file} fayli muvaffaqiyatli yaratildi.")
