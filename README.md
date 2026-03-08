# EmotienSense — YOLOv8

A web app for real-time facial emotion recognition, built with YOLOv8 and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFCF)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is this?

EmotienSense lets you upload a photo and instantly see which emotion is showing on the face in it. It's powered by a YOLOv8 classification model trained on facial images, and the whole thing runs in a Streamlit interface.

I built this as a personal project to explore emotion recognition with computer vision. It detects the 7 basic emotions from Ekman's model — happiness, sadness, anger, fear, disgust, surprise, and neutral — and shows you both the top result and a breakdown of all confidence scores.

The UI has a dark and light theme, and you can analyze images one at a time or upload a batch of them.

---

## Getting started

You'll need Python 3.9 or newer. If you're on Linux, also install these system packages first:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

Clone the repo, set up a virtual environment, then install the dependencies:

```bash
git clone https://github.com/Obelisk999/EmotienSense-YOLOv8.git
cd EmotienSense-YOLOv8

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## Running the app

```bash
streamlit run streamlit-app.py
```

Open `http://localhost:8501` in your browser. Pick single image or batch mode from the sidebar, upload your file(s), and that's it.

---

## Emotions it can detect

The model classifies faces into these 7 categories:

- 😲 Surprise
- 😨 Fear
- 🤢 Disgust
- 😄 Happiness
- 😢 Sadness
- 😠 Anger
- 😐 Neutral

---

## How it works

Images get resized to 64×64 and fed into a YOLOv8 classification model (`best.pt`). If the `.pt` file isn't available, it falls back to the ONNX version (`best.onnx`). The model outputs confidence scores for each emotion class, and the top result is shown alongside a probability bar chart.

The model is cached with `@st.cache_resource` so it only loads once per session.

---

## Project files

```
EmotienSense-YOLOv8/
├── streamlit-app.py   # main app
├── best.pt            # YOLOv8 weights (PyTorch)
├── best.onnx          # exported ONNX weights
├── requirements.txt
├── packages.txt       # system deps for Streamlit Cloud
└── README.md
```

---

## Contributing

PRs are welcome. Just fork the repo, make your changes on a new branch, and open a pull request. No strict conventions, just keep it readable.

---

## License

MIT — see [LICENSE](LICENSE).