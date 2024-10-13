from flask import Flask, render_template, Response, jsonify
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

app = Flask(__name__)

# Daftar URL CCTV yang ingin dipantau
urls = [
    "https://cctv.balitower.co.id/Monas-Barat-009-506632_2/embed.html",
    "https://cctv.balitower.co.id/Menteng-001-700123_5/embed.html",
    "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_3/embed.html",
    "https://cctv.balitower.co.id/Gelora-017-700470_2/embed.html",
]

# Daftar untuk menyimpan link stream
stream_links = []

# Fungsi untuk mendapatkan link stream dari CCTV menggunakan Selenium
def get_stream_links():
    global stream_links
    stream_links = []  # Reset daftar link
    driver_service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=driver_service)

    for url in urls:
        driver.get(url)
        time.sleep(5)  # Tunggu agar halaman dimuat

        try:
            iframe = driver.find_element(By.TAG_NAME, "iframe")
            stream_link = iframe.get_attribute("src")
            print(f"Link stream ditemukan: {stream_link}")
            stream_links.append(stream_link)
        except Exception as e:
            print("Tidak ditemukan iframe, coba mencari video.")
            try:
                video = driver.find_element(By.TAG_NAME, "video")
                stream_link = video.get_attribute("src")
                print(f"Link stream ditemukan: {stream_link}")
                stream_links.append(stream_link)
            except Exception as e:
                print("Tidak ada video.")
                stream_links.append(None)

    driver.quit()

# Daftar untuk menyimpan jumlah orang dan mobil yang terdeteksi di setiap CCTV
person_count_dict = {i: {'person': 0, 'car': 0} for i in range(len(urls))}

# Fungsi untuk menampilkan video stream dan mendeteksi orang serta mobil menggunakan YOLOv8
def generate_frames(cctv_id):
    global person_count_dict
    stream_link = stream_links[cctv_id]
    model = YOLO('D:/GEMASTIK 2025 MACHINE LEARNING/yolov5/yolov8n.pt')  # Path ke YOLOv8

    cap = cv2.VideoCapture(stream_link)

    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka stream {cctv_id}")
        return

    person_count = 0
    car_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print(f"Stream {cctv_id} terputus, mencoba lagi...")
            time.sleep(5)
            cap = cv2.VideoCapture(stream_link)
            continue

        frame_resized = cv2.resize(frame, (640, 360))  # Mengubah ukuran frame

        # Deteksi objek menggunakan YOLOv8
        results = model(frame_resized)

        person_count = 0
        car_count = 0
        if results:
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = ""
                        color = (0, 255, 0)  # Warna default (hijau)

                        if cls_id == 0:  # Person class
                            person_count += 1
                            label = "Person"
                            color = (0, 255, 0)  # Hijau
                        elif cls_id == 2:  # Car class (periksa ID untuk mobil)
                            car_count += 1
                            label = "Car"
                            color = (255, 0, 0)  # Biru

                        if label:
                            # Menggambar bounding box tipis (tebal 1)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)
                            cv2.putText(frame_resized, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Simpan jumlah orang dan mobil ke dalam dictionary
        person_count_dict[cctv_id] = {'person': person_count, 'car': car_count}

        # Encode frame sebagai JPEG
        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Mendapatkan link stream saat halaman utama dibuka
    get_stream_links()
    return render_template('index copy.html')

@app.route('/video_feed/<int:cctv_id>')
def video_feed(cctv_id):
    # Menghasilkan video stream untuk setiap CCTV
    return Response(generate_frames(cctv_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/person_count')
def person_count():
    # Mengembalikan jumlah orang dan mobil dalam bentuk JSON
    return jsonify(person_count_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
