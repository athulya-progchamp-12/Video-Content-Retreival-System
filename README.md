# InVideoSearch

**Video Content Retrieval System** (VidExplore / InVideoSearch)

This repository contains the project files for a Video Content Retrieval System developed as a final-year project. The system extracts keyframes from video, generates captions, and performs semantic search to retrieve relevant video segments based on textual queries.

## Included
- `vqa_smolvlm_streamlitfinalpart1(1).ipynb` : Original Colab/Jupyter notebook (uploaded by the author)
- `finalfinalproject.pptx` : Project presentation slides
- `main.py` : Auto-generated script compiled from code cells in the notebook
- `requirements.txt` : Python dependencies
- `data/` : Folder for sample video or datasets
- `outputs/frames/` : Suggested place for saved keyframes
- `models/` : Place to store any downloaded/fine-tuned models

## How to use
1. Clone the repository or unzip the package.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook with Jupyter/Colab or run `main.py` (script generated from the notebook code cells).
4. Put videos in `data/` and outputs will be saved under `outputs/` by default.

## Notes
- The notebook is included as the primary source of code and experiments. `main.py` is a convenience script generated from notebook code cells; please review it before running.
- Large videos are not included. Add videos to `data/` or update `.gitignore` to exclude outputs when pushing to GitHub.

---
Project automated package created for the student. Good luck!
