import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import requests
import tempfile
import os

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "yolov8n.pt"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 🔐 Secure

model = YOLO(MODEL_PATH)

# -------------------------------
# LLM FUNCTION (FIXED + SAFE)
# -------------------------------
def generate_llm_response(text):
    if not GROQ_API_KEY:
        return "⚠️ API Key not set."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a traffic analysis assistant."},
            {"role": "user", "content": f"Explain this traffic detection: {text}"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "⚠️ AI unavailable (API error)"

    except:
        # ✅ Offline fallback
        return f"📊 Detected: {text}. Follow traffic safety rules."

# -------------------------------
# TEXT TO SPEECH (SAFE)
# -------------------------------
def text_to_speech(text):
    try:
        tts = gTTS(text)
        file_path = "output.mp3"
        tts.save(file_path)
        return file_path
    except:
        return None

# -------------------------------
# IMAGE DETECTION
# -------------------------------
def detect_image(image):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    results = model(image)
    detections = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "type": label,
                "confidence": round(conf, 2),
                "location": (x1, y1, x2, y2)
            })

            cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image, f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    return image, detections

# -------------------------------
# VIDEO DETECTION
# -------------------------------
def detect_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "type": label,
                    "confidence": round(conf, 2),
                    "location": (x1, y1, x2, y2)
                })

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{label} {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        stframe.image(frame, channels="BGR")

    cap.release()

    unique = {str(d): d for d in detections}
    return list(unique.values())

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

st.title("🚦 Adaptive Traffic Intelligence System")

option = st.radio("Choose Input Type", ["Image", "Video"])

# -------------------------------
# IMAGE
# -------------------------------
if option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        image = np.array(image)

        st.image(image)

        if st.button("Detect Image"):
            img, det = detect_image(image.copy())
            st.image(img)

            if det:
                st.subheader("📊 Results")

                summary = []
                for d in det:
                    st.write(d)
                    summary.append(f"{d['type']} ({d['confidence']})")

                text = ", ".join(summary)

                st.subheader("🧠 AI Explanation")
                st.write(generate_llm_response(text))

                st.subheader("🔊 Audio")
                audio = text_to_speech(text)
                if audio:
                    st.audio(audio)
                else:
                    st.warning("Audio unavailable")

# -------------------------------
# VIDEO
# -------------------------------
elif option == "Video":
    file = st.file_uploader("Upload Video", type=["mp4","avi"])

    if file:
        st.video(file)

        if st.button("Detect Video"):
            det = detect_video(file)

            if det:
                st.subheader("📊 Results")

                summary = []
                for d in det:
                    st.write(d)
                    summary.append(f"{d['type']} ({d['confidence']})")

                text = ", ".join(summary)

                st.subheader("🧠 AI Explanation")
                st.write(generate_llm_response(text))

                st.subheader("🔊 Audio")
                audio = text_to_speech(text)
                if audio:
                    st.audio(audio)
                else:
                    st.warning("Audio unavailable")
