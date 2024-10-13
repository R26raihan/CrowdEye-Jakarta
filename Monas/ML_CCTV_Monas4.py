from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import cv2
import time
from ultralytics import YOLO
import os

# Ganti dengan path ke chromedriver yang sudah diunduh dan diekstrak
chrome_driver_path = "D:/GEMASTIK 2025 MACHINE LEARNING/chromedriver-win64/chromedriver.exe"

# Menambahkan environment variable untuk mengatasi OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Inisialisasi Service untuk Selenium
driver_service = Service(chrome_driver_path)

# Inisialisasi driver Chrome
driver = webdriver.Chrome(service=driver_service)

# URL CCTV yang ingin diambil
urls = [
    "https://cctv.balitower.co.id/Monas-Barat-004_a/embed.html",
]

stream_links = []

for url in urls:
    # Buka halaman CCTV
    driver.get(url)

    # Tunggu hingga elemen <iframe> atau <video> terdeteksi (timeout 15 detik)
    time.sleep(5)

    try:
        iframe = driver.find_element(By.TAG_NAME, "iframe")
        stream_link = iframe.get_attribute("src")
        print(f"Link stream ditemukan: {stream_link}")
        stream_links.append(stream_link)
    except Exception as e:
        print("Tidak ditemukan iframe di halaman, coba mencari elemen video langsung.")
        print(f"Error: {e}")

        try:
            video = driver.find_element(By.TAG_NAME, "video")
            stream_link = video.get_attribute("src")
            print(f"Link stream ditemukan: {stream_link}")
            stream_links.append(stream_link)
        except Exception as e:
            print("Tidak ditemukan video di halaman.")
            print(f"Error: {e}")
            stream_links.append(None)

# Tutup browser setelah link ditemukan
driver.quit()

# Menampilkan video dari link yang ditemukan
if not all(stream_links):
    print("Salah satu atau lebih link stream tidak ditemukan, program dihentikan.")
else:
    print("Mencoba membuka link stream:")
    cap = cv2.VideoCapture(stream_links[0])  # Menggunakan stream pertama

    # Periksa apakah video capture berhasil dibuka
    if not cap.isOpened():
        print("Error: Tidak dapat membuka stream video.")
    else:
        print("Sukses: Stream video berhasil dibuka.")

        # Load YOLOv8 model
        model = YOLO('D:\GEMASTIK 2025 MACHINE LEARNING\yolov5\yolov8n.pt')

        # Loop untuk membaca dan memproses frame
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Stream terputus, mencoba untuk membuka kembali...")
                time.sleep(5)
                cap = cv2.VideoCapture(stream_links[0])  # Mencoba untuk membuka kembali stream
                continue

            # Mengubah ukuran frame
            frame_resized = cv2.resize(frame, (640, 360))

            # Deteksi objek
            results = model.track(frame_resized, persist=True, conf=0.3)

            # Hitung jumlah orang
            person_count = 0
            if results:
                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            class_id = box.cls[0].item()  # Ambil ID kelas
                            if class_id == 0:  # Jika kelas adalah 'person'
                                person_count += 1
            
            # Tampilkan jumlah orang di frame
            cv2.putText(frame_resized, f'Jumlah Orang: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Plot hasil deteksi
            frame_ = results[0].plot() if results else frame_resized

            # Tampilkan frame yang sudah diproses
            cv2.imshow('CCTV Stream', frame_)

            # Tunggu sebentar untuk menjaga frame rate
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
