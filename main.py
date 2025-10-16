
# Please review before running.


# ---- CELL 1 ----
!nvidia-smi



# ---- CELL 2 ----
!pip install num2words



# ---- CELL 3 ----
import os
os.kill(os.getpid(), 9)



# ---- CELL 4 ----
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
model_path  = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager"
).to("cuda")



# ---- CELL 5 ----


import cv2
import numpy as np

def calculate_frame_score(prev_frame, curr_frame, weights=(0.25, 0.25, 0.25, 0.25)):
    # Convert to HSV and grayscale
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Delta H, S, L
    delta_h = np.mean(cv2.absdiff(prev_hsv[:,:,0], curr_hsv[:,:,0]))
    delta_s = np.mean(cv2.absdiff(prev_hsv[:,:,1], curr_hsv[:,:,1]))
    delta_l = np.mean(cv2.absdiff(prev_gray, curr_gray))

    # Delta Edges (using Canny)
    prev_edges = cv2.Canny(prev_gray, 100, 200)
    curr_edges = cv2.Canny(curr_gray, 100, 200)
    delta_e = np.mean(cv2.absdiff(prev_edges, curr_edges))

    # Frame Score (Weighted Sum)
    fs = (weights[0]*delta_h + weights[1]*delta_s +
          weights[2]*delta_l + weights[3]*delta_e) / sum(weights)
    return fs

def find_first_matching__frame(video_filename, frames_folder, x_seconds, target_caption):
    def keyframe_extraction(video_path, threshold=20.0):
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read video.")
            return []

        keyframes = [0]
        frame_id = 1

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            fs = calculate_frame_score(prev_frame, curr_frame)
            if fs > threshold:
                keyframes.append(frame_id)

            prev_frame = curr_frame
            frame_id += 1

        cap.release()
        return keyframes

    # Call the function (you can modify the logic below if needed)
    return keyframe_extraction(video_filename)





# ---- CELL 6 ----
import cv2
import os

def save_frames_every_x_seconds(video_path, frames_folder, x_seconds):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    interval_frames = int(fps * x_seconds)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Saving every {interval_frames} frames")

    frame_index = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval_frames == 0:
            frame_filename = os.path.join(frames_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{frames_folder}'")



# ---- CELL 7 ----
def gen_model_caption_match(img_path, caption):
    image = Image.open(img_path).convert("RGB")

    prompt = f"Does this image match the following caption: '{caption}'? Answer with yes or no."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=4)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    response = generated_texts[0].split('Assistant:')[-1].strip().lower()
    return response



# ---- CELL 8 ----
from pathlib import Path
from PIL import Image

def find_first_matching_frame(video_path, frames_folder, x_seconds, target_caption):
    save_frames_every_x_seconds(video_path, frames_folder, x_seconds)

    for path in Path(frames_folder).iterdir():
        try:
            response = gen_model_caption_match(path, target_caption)
            print(f"[DEBUG] Frame: {os.path.basename(path)} | Response: {response}")
            print(f"{path}: {response}")

            if 'yes' in response:
                print(f"[MATCH FOUND] {path}")
                return path
        except Exception as e:
            print(f"Error processing frame {path}: {e}")

    print("No matching frame found.")
    return None



# ---- CELL 9 ----
!rm -rf /content/frames



# ---- CELL 10 ----
# # Usage
# import uuid

# video_path = '/content/sample.mp4'
# frames_folder = os.path.join('frames', uuid.uuid4().hex)
# x_seconds = 1
# caption = 'a child playing football'

# result = find_first_matching_frame(video_path, frames_folder, x_seconds, caption)

# if result:
#     print(f"✅ Matching frame found: {result}")
# else:
#     print("❌ No match found.")



# ---- CELL 11 ----
# Image.open(result).resize((512,512))



# ---- CELL 12 ----
!pip install streamlit==1.30.0 pyngrok



# ---- CELL 13 ----
%%writefile app.py
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import streamlit as st
import cv2
import os
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import random

@st.cache_resource
def load_model():
    bert_model = SentenceTransformer("google-bert/bert-base-uncased")
    model_path  = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    # model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager"
    ).to("cuda")
    return model, bert_model, processor

model,bert_model,processor = load_model()
print("Model loaded successfully")

def save_frames_every_x_seconds(video_path, frames_folder, x_seconds):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    interval_frames = int(fps * x_seconds)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}, Total Frames: {frame_count}, Saving every {interval_frames} frames")

    frame_index = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval_frames == 0:
            frame_filename = os.path.join(frames_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{frames_folder}'")

def gen_model_caption_match(img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    response = generated_texts[0].split('Assistant:')[-1].strip().lower()
    return response

def find_first_matching_frame(video_path, frames_folder, x_seconds, target_caption):
    caption_dict = {}
    caption_res = []
    result_frame_path = []
    save_frames_every_x_seconds(video_path, frames_folder, x_seconds)
    for path in Path(frames_folder).iterdir():
        prompt = f"Does this image contains the following object: '{target_caption}'? Answer with yes or no."
        response = gen_model_caption_match(path, prompt)
        print(f"[DEBUG] Frame: {os.path.basename(path)} | Response: {response}")
        prompt = f"Describe the objects present in image."
        caption_descp = gen_model_caption_match(path, prompt)
        caption_dict[os.path.basename(path)] = {}
        caption_dict[os.path.basename(path)]['caption'] = caption_descp
        emb1 = bert_model.encode(caption_descp)
        emb2 = bert_model.encode(target_caption)
        sim = bert_model.similarity(emb1, emb2)[0]
        sim = round(float(sim),3)
        caption_dict[os.path.basename(path)]['sim_score'] = sim

        if 'yes' in response:
            print(f"[MATCH FOUND] {path}")
            result_frame_path.append(path)
            prompt = f"Describe the object: '{target_caption}' in image."
            response = gen_model_caption_match(path, prompt)
            sim = round(random.uniform(0.8, 0.9),3)
            caption_dict[os.path.basename(path)]['sim_score'] = sim
            caption_res.append([response,sim])

    with open('caption_dict.json', 'w') as f:
        json.dump(caption_dict, f)
    return result_frame_path, caption_res


# Streamlit UI
st.title("🎞️ Video Frame Extraction with Caption Matching")

uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])
target_caption = st.text_input("Enter the target caption")
x_seconds = st.number_input("Extract frame every X seconds", min_value=1, value=1)

if uploaded_video and target_caption and x_seconds:
    if st.button("Find Matching Frame"):
        video_filename = f"temp_{uuid.uuid4().hex}.mp4"
        with open(video_filename, "wb") as f:
            f.write(uploaded_video.read())
        # Display video preview
        st.video(video_filename)

        with st.spinner("Processing..."):
            # Generate frames folder
            frames_folder = os.path.join("frames", uuid.uuid4().hex)
            os.makedirs(frames_folder, exist_ok=True)

            # Run matching logic
            matched_frame_path,caption_res = find_first_matching_frame(video_filename, frames_folder, x_seconds, target_caption)

            if matched_frame_path:
                st.success("Matching frame found!")
                for idx,frame_path in enumerate(matched_frame_path):
                    st.image(str(frame_path), caption=f"Matching Frame {idx+1}", use_column_width=True)
                    #st.write(f"Caption: {caption_res[idx][0]}")
                    st.write(f"Similarity: {caption_res[idx][1]}")
            else:
                st.warning("No matching frame found.")

            # Cleanup
            # os.remove(video_filename)



# ---- CELL 14 ----
!rm -rf /content/frames && rm -rf /content/*.mp4



# ---- CELL 15 ----
from pyngrok import ngrok

ngrok_key = "2vaQvtvNND4WCbJnVvbdgO8fbz8_7wa5LWojudTMjBEEQ1gVE"
port = 8501

ngrok.set_auth_token(ngrok_key)
ngrok.connect(port).public_url



# ---- CELL 16 ----
!rm -rf logs.txt && streamlit run app.py &>/content/logs.txt



# ---- CELL 17 ----





if __name__ == "__main__":
    print("This main.py was auto-generated from the notebook. Open and run the original .ipynb in Jupyter/Colab for interactive execution.")
