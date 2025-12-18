# 🚀 Quick Start Guide - Sign Language Translator

Get the application running in 10 minutes!

---

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Webcam** (for sign language detection)
- **Microphone** (optional, for speech input)
- **Internet connection** (for initial setup)

---

## ⚡ Super Quick Start (5 Minutes)

### 1. Clone Repository
```bash
git clone https://github.com/Akrati36/sign-language-translator.git
cd sign-language-translator
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

**That's it!** Application opens at http://localhost:8501

---

## 🎯 Detailed Setup

### Step 1: System Requirements

**Check Python version:**
```bash
python --version
# Should be 3.8 or higher
```

**Check webcam:**
- Make sure your webcam is connected
- Test it with your system's camera app

### Step 2: Clone & Setup

```bash
# Clone repository
git clone https://github.com/Akrati36/sign-language-translator.git
cd sign-language-translator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# This will install:
# - Streamlit (web interface)
# - OpenCV (computer vision)
# - MediaPipe (hand detection)
# - TensorFlow (deep learning)
# - SpeechRecognition (speech-to-text)
# - pyttsx3 (text-to-speech)
# - And more...
```

**Note for PyAudio (Speech Recognition):**

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Step 4: Download Models (Optional)

```bash
# Download pre-trained models
python download_models.py

# This downloads:
# - Sign classifier model
# - Hand gesture detector
# - Configuration files
```

**Note:** If models aren't available, the app will use a demo mode.

### Step 5: Run Application

```bash
# Run Streamlit app
streamlit run app.py

# Application will open in your browser at:
# http://localhost:8501
```

---

## 🎮 Using the Application

### Mode 1: Sign Language to Text

1. **Click "Start Camera"**
2. **Allow camera access** when prompted
3. **Show sign language gestures** to webcam
4. **See real-time translation** on screen
5. **Click "Speak"** to hear the text

**Tips:**
- Keep hand in frame
- Good lighting helps
- Clear background works best
- Hold each sign for 1-2 seconds

### Mode 2: Text to Sign Language

1. **Type your message** in text box
2. **Or click "Speak"** and say your message
3. **Click "Translate"**
4. **See sign language animations**

### Mode 3: Learning

1. **Select lesson type** (Alphabet, Numbers, Words)
2. **Watch tutorial videos**
3. **Practice with webcam**
4. **Get instant feedback**

### Mode 4: Emergency

1. **Click emergency phrase**
2. **System shows sign**
3. **Speaks the phrase**
4. **Sends alert** (if configured)

---

## 🔧 Troubleshooting

### Camera Not Working

**Problem:** "Failed to access camera"

**Solutions:**
1. Check camera is connected
2. Close other apps using camera
3. Grant camera permissions
4. Try different camera index:
   ```python
   # In app.py, change:
   cap = cv2.VideoCapture(0)  # Try 1, 2, etc.
   ```

### Import Errors

**Problem:** "ModuleNotFoundError"

**Solution:**
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

### Speech Recognition Not Working

**Problem:** "Could not understand audio"

**Solutions:**
1. Check microphone is connected
2. Speak clearly and loudly
3. Reduce background noise
4. Check microphone permissions

### Slow Performance

**Problem:** App is laggy

**Solutions:**
1. Close other applications
2. Reduce video quality
3. Use GPU if available
4. Lower confidence threshold

---

## 📊 Testing the Application

### Test Sign Detection

```bash
# Run test script
python tests/test_detector.py

# Should output:
# ✓ Hand detector initialized
# ✓ Detected 2 hands
# ✓ Extracted 21 landmarks
```

### Test Classification

```bash
# Run classifier test
python tests/test_classifier.py

# Should output:
# ✓ Model loaded
# ✓ Prediction: A (confidence: 0.96)
```

---

## 🎯 Next Steps

### Customize Settings

Edit `config.json`:
```json
{
  "confidence_threshold": 0.7,
  "max_hands": 2,
  "language": "ASL",
  "enable_speech": true,
  "speech_rate": 150
}
```

### Add More Signs

Add to `data/signs_database.json`:
```json
{
  "new_word": {
    "word": "new_word",
    "type": "word",
    "description": "How to sign it",
    "animation": "path/to/animation.gif"
  }
}
```

### Train Your Own Model

```bash
# Collect data
python scripts/collect_data.py

# Train model
python scripts/train_model.py

# Evaluate
python scripts/evaluate_model.py
```

---

## 🆘 Getting Help

**If you're stuck:**

1. **Check documentation:** [README.md](README.md)
2. **Search issues:** [GitHub Issues](https://github.com/Akrati36/sign-language-translator/issues)
3. **Ask community:** [Discord](https://discord.gg/signlang)
4. **Email support:** support@signlanguagetranslator.com

---

## ✅ Checklist

Before you start:
- [ ] Python 3.8+ installed
- [ ] Webcam connected
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Application runs successfully

---

## 🎉 You're Ready!

**Start translating sign language!**

**Help make communication accessible to everyone! 🤟**

---

**Questions?** Open an issue on GitHub!

**Found a bug?** Report it!

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md)!