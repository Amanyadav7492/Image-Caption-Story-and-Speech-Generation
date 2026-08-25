# Image Caption Generator

An AI-powered image captioning system that generates descriptive captions for images using deep learning. Captions are generated with BLIP, then a short story is created from the caption + optional user keywords.

## Features

- **Deep Learning Model**: Uses BLIP for image caption generation
- **Web Interface**: Streamlit-based web application for image upload, caption generation, and story generation
- **Keywords input**: Provide keywords via text and/or microphone speech
- **Text-to-speech**: Listen to the generated story as audio
- **Pre-trained Models**: Uses a pre-trained BLIP model (downloaded automatically by `transformers`)
- **Jupyter Notebook**: Complete training pipeline with data visualization and model evaluation

## Project Structure

```
IMAGE_cation./
├── Images/                          # Dataset images (8,091 JPG files)
├── models/                          # Trained models and tokenizer
│   ├── feature_extractor.keras      # DenseNet201 feature extractor
│   ├── model.keras                  # Main caption generation model
│   └── tokenizer.pkl               # Text tokenizer for captions
├── image-caption.ipynb             # Jupyter notebook with training code
├── main.py                         # Streamlit web application
├── captions.txt                    # Image captions dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd IMAGE_cation.
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application (Streamlit)

Run the Streamlit web application for an interactive interface:

```bash
streamlit run main.py
```

The application will open in your browser where you can:
- Upload any image (JPG, JPEG, PNG)
- (Optional) Enter keywords in a text box
- (Optional) Speak keywords using your microphone (speech-to-text)
- Generate and view:
  - An image caption
  - A single-paragraph story (50–60 words)
  - Audio playback of the story (text-to-speech)

Notes:
- **Speech-to-text** currently uses `SpeechRecognition` with Google's recognizer, which typically **requires internet access**.
- **Text-to-speech** uses `gTTS` (Google Text-to-Speech), which also typically **requires internet access**.

### Jupyter Notebook

Open the Jupyter notebook to explore the training process:

```bash
jupyter notebook image-caption.ipynb
```

The notebook contains:
- Data preprocessing and visualization
- Model architecture definition
- Training pipeline with callbacks
- Model evaluation and testing

### Direct Python Usage

The Streamlit app is the recommended way to run this project.

## Model Architecture

The web app uses:

- **Image captioning**: BLIP (pre-trained transformer) to generate a caption from the uploaded image.
- **Story generation**: BART (pre-trained transformer) generates an action-focused one-paragraph story from the caption + optional keywords, then enforces the target word range.

## Dataset

The model was trained on the Flickr8K dataset which includes:
- 8,091 images
- 5 captions per image
- Preprocessed text with start/end sequence tokens

## Training Details

The Streamlit app now uses:
- **Image captioning**: BLIP (pre-trained transformer)
- **Story generation**: BART (pre-trained transformer) with enforced 50–60 word output

The original Jupyter notebook still contains the older CNN+LSTM training pipeline for experimentation.

## Requirements

- Python 3.8+
- TensorFlow 2.15.0+
- Streamlit 1.36.0+
- Other dependencies listed in `requirements.txt`

## Performance

The model achieves good performance on image captioning tasks with:
- Reasonable caption quality for most images
- Fast inference time
- Support for various image formats

## Limitations

- Caption quality depends on training data
- May not perform well on images very different from training set
- Limited to English captions only
- Requires significant computational resources for training

## Contributing

Feel free to contribute to this project by:
- Improving model architecture
- Adding new features to the web interface
- Optimizing performance
- Adding support for more languages

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Flickr8K dataset creators
- TensorFlow and Keras teams
- Streamlit for the web interface
- The open-source community for various libraries used
