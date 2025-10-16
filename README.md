# InVideoSearch

**Video Content Retrieval System**

This repository contains the project files for a Video Content Retrieval System developed as a final-year project. The system extracts keyframes from video, generates captions, and performs semantic search to retrieve relevant video segments based on textual queries.

## Features

- Upload a video file (`mp4`)
- Extract frames every X seconds  
- Detect frames matching a target caption  
- Display matching frames with similarity scores  
- Run Streamlit app inside Colab using ngrok  

## Quick Start in Google Colab

1. Open this notebook in Colab: [Open in Colab](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/<your-notebook>.ipynb)  
2. Run **all cells sequentially**.  
3. If prompted, enter your **ngrok authentication key** (already included if present).  
4. Upload your video using the Streamlit uploader.  
5. Enter the **target caption** and **frame interval** (seconds).  
6. Click **Find Matching Frame**.  

The notebook will display matching frames and similarity scores.

---

## Optional: Run Locally

If you want to run the app outside Colab:

1. Install Python dependencies:

```bash
pip install -r requirements.txt

2.Run the streamlit app
streamlit run app.py

## Notes

Outputs have been cleared for GitHub rendering.
Use a GPU runtime in Colab for faster processing.
