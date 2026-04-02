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
MODEL_PATH = "yolov8n.pt"   # Replace with traffic_sign_model.pt later
GROQ_API_KEY = "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"

model = YOLO(MODEL_PATH)

# -------------------------------
# LLM RESPONSE (GROQ)
# -------------------------------
def generate_llm_response(text):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gemma2-9b-it",
            "messages": [
                {"role": "user", "content": f"Explain this traffic detection clearly: {text}"}
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"LLM Error: {response.status_code}"

    except Exception as e:
        return f"Connection Error: {e}"

# -------------------------------
# TEXT TO SPEECH
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
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Store structured data
            detections.append({
                "type": label,
                "confidence": round(conf, 2),
                "location": (x1, y1, x2, y2)
            })

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    return image, detections

# -------------------------------
# VIDEO DETECTION
# -------------------------------
def detect_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detection = {
                    "type": label,
                    "confidence": round(conf, 2),
                    "location": (x1, y1, x2, y2)
                }

                all_detections.append(detection)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

    # Remove duplicates
    unique = {str(d): d for d in all_detections}
    return list(unique.values())

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

st.title("🚦 Adaptive Traffic Intelligence System")

option = st.radio("Choose Input Type", ["Image", "Video"])

# -------------------------------
# IMAGE SECTION
# -------------------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

        st.image(image, caption="Uploaded Image")

        if st.button("Detect Image"):
            result_img, detections = detect_image(image.copy())

            st.image(result_img, caption="Detected Image")

            if detections:
                st.subheader("📊 Detection Results")

                summary_text = []

                for d in detections:
                    st.write(f"🔹 Type: {d['type']}")
                    st.write(f"🔹 Confidence: {d['confidence']}")
                    st.write(f"🔹 Location: {d['location']}")
                    st.write("---")

                    summary_text.append(f"{d['type']} ({d['confidence']})")

                result_text = ", ".join(summary_text)

                # LLM
                st.subheader("🧠 AI Explanation")
                explanation = generate_llm_response(result_text)
                st.write(explanation)

                # AUDIO
                st.subheader("🔊 Audio Output")
                audio_file = text_to_speech(result_text)
                if audio_file:
                    st.audio(audio_file)
                else:
                    st.warning("Audio generation failed")

            else:
                st.warning("No objects detected")

# -------------------------------
# VIDEO SECTION
# -------------------------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Detect Video"):
            detections = detect_video(uploaded_video)

            if detections:
                st.subheader("📊 Detection Results")

                summary_text = []

                for d in detections:
                    st.write(f"🔹 Type: {d['type']}")
                    st.write(f"🔹 Confidence: {d['confidence']}")
                    st.write(f"🔹 Location: {d['location']}")
                    st.write("---")

                    summary_text.append(f"{d['type']} ({d['confidence']})")

                result_text = ", ".join(summary_text)

                st.subheader("🧠 AI Explanation")
                st.write(generate_llm_response(result_text))

                st.subheader("🔊 Audio Output")
                audio_file = text_to_speech(result_text)
                if audio_file:
                    st.audio(audio_file)

            else:
                st.warning("No objects detected")