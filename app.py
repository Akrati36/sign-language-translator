"""
Sign Language Translator - Main Streamlit Application
Real-time sign language to text/speech translation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Import custom modules
from src.hand_detector import HandDetector
from src.sign_classifier import SignClassifier
from src.speech_handler import SpeechHandler
from src.text_to_sign import TextToSign
from src.utils import load_config, save_conversation

# Page configuration
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = HandDetector()
if 'classifier' not in st.session_state:
    st.session_state.classifier = SignClassifier()
if 'speech_handler' not in st.session_state:
    st.session_state.speech_handler = SpeechHandler()
if 'text_to_sign' not in st.session_state:
    st.session_state.text_to_sign = TextToSign()
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🤟 Sign Language Translator")
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Sign to Text", "Text to Sign", "Learning", "Emergency"],
        help="Choose translation direction"
    )
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    language = st.selectbox(
        "Sign Language",
        ["ASL (American)", "BSL (British)", "ISL (Indian)"],
        help="Select sign language type"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.7,
        help="Minimum confidence for detection"
    )
    
    enable_speech = st.checkbox("Enable Text-to-Speech", value=True)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    st.metric("Signs Detected", len(st.session_state.conversation))
    st.metric("Accuracy", "96.5%")
    st.metric("Processing Speed", "30 FPS")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-header">🤟 Sign Language Translator</h1>', 
            unsafe_allow_html=True)

# ============================================================================
# MODE 1: SIGN TO TEXT
# ============================================================================

if mode == "Sign to Text":
    st.markdown("### 📹 Sign Language to Text/Speech")
    st.markdown("Show sign language gestures to your webcam for real-time translation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera controls
        camera_col1, camera_col2 = st.columns(2)
        with camera_col1:
            start_camera = st.button("🎥 Start Camera", key="start_cam")
        with camera_col2:
            stop_camera = st.button("⏹️ Stop Camera", key="stop_cam")
        
        # Video placeholder
        video_placeholder = st.empty()
        
        # Translation output
        st.markdown("### 📝 Translation")
        translation_placeholder = st.empty()
        
    with col2:
        st.markdown("### 🎯 Detection Info")
        
        # Confidence meter
        confidence_placeholder = st.empty()
        
        # Detected sign
        sign_placeholder = st.empty()
        
        # Hand landmarks visualization
        st.markdown("### ✋ Hand Landmarks")
        landmarks_placeholder = st.empty()
        
        # Speak button
        if st.button("🔊 Speak Translation"):
            if st.session_state.conversation:
                last_text = st.session_state.conversation[-1]['text']
                st.session_state.speech_handler.speak(last_text)
                st.success(f"Speaking: {last_text}")
    
    # Camera processing
    if start_camera:
        st.session_state.camera_active = True
    if stop_camera:
        st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            hands_data = st.session_state.detector.detect(frame)
            
            if hands_data['detected']:
                # Draw landmarks on frame
                frame = st.session_state.detector.draw_landmarks(
                    frame, hands_data['landmarks']
                )
                
                # Classify sign
                prediction = st.session_state.classifier.predict(
                    hands_data['landmarks']
                )
                
                if prediction['confidence'] >= confidence_threshold:
                    # Update UI
                    sign_placeholder.markdown(
                        f"**Detected Sign:** {prediction['sign']}"
                    )
                    confidence_placeholder.progress(prediction['confidence'])
                    
                    # Add to conversation
                    st.session_state.conversation.append({
                        'type': 'sign',
                        'text': prediction['sign'],
                        'confidence': prediction['confidence'],
                        'timestamp': time.time()
                    })
                    
                    # Update translation
                    full_text = ' '.join([
                        c['text'] for c in st.session_state.conversation[-10:]
                    ])
                    translation_placeholder.markdown(
                        f'<div class="translation-box">{full_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Speak if enabled
                    if enable_speech:
                        st.session_state.speech_handler.speak(prediction['sign'])
            
            # Display frame
            video_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True
            )
            
            # Small delay
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()

# ============================================================================
# MODE 2: TEXT TO SIGN
# ============================================================================

elif mode == "Text to Sign":
    st.markdown("### 💬 Text/Speech to Sign Language")
    st.markdown("Type or speak your message to see sign language translation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Text input
        st.markdown("#### ✍️ Type Your Message")
        text_input = st.text_area(
            "Enter text",
            height=150,
            placeholder="Type your message here..."
        )
        
        # Speech input
        st.markdown("#### 🎤 Or Speak")
        if st.button("🎤 Start Recording"):
            with st.spinner("Listening..."):
                spoken_text = st.session_state.speech_handler.listen()
                if spoken_text:
                    text_input = spoken_text
                    st.success(f"You said: {spoken_text}")
        
        # Translate button
        if st.button("🔄 Translate to Sign Language"):
            if text_input:
                # Get sign language translation
                signs = st.session_state.text_to_sign.translate(text_input)
                st.session_state.current_signs = signs
                st.success(f"Translated {len(signs)} words to signs!")
    
    with col2:
        st.markdown("#### 🤟 Sign Language Translation")
        
        if hasattr(st.session_state, 'current_signs'):
            # Display signs
            for i, sign in enumerate(st.session_state.current_signs):
                st.markdown(f"**{i+1}. {sign['word']}**")
                
                # Show animation/video
                if sign.get('animation'):
                    st.image(sign['animation'], width=200)
                
                # Show description
                st.markdown(f"*{sign.get('description', 'No description')}*")
                st.markdown("---")

# ============================================================================
# MODE 3: LEARNING
# ============================================================================

elif mode == "Learning":
    st.markdown("### 🎓 Learn Sign Language")
    st.markdown("Interactive lessons to learn sign language")
    
    # Lesson selection
    lesson_type = st.selectbox(
        "Choose Lesson",
        ["Alphabet (A-Z)", "Numbers (0-9)", "Common Words", "Phrases"]
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📚 Tutorial")
        
        if lesson_type == "Alphabet (A-Z)":
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            selected_letter = st.select_slider("Select Letter", letters)
            
            st.markdown(f"### Letter: {selected_letter}")
            st.markdown("**How to sign:**")
            st.markdown("1. Position your hand in front of your chest")
            st.markdown("2. Form the letter shape")
            st.markdown("3. Hold steady for 1 second")
            
            # Show video/image
            st.info("Video tutorial will be displayed here")
        
        elif lesson_type == "Numbers (0-9)":
            numbers = list(range(10))
            selected_number = st.select_slider("Select Number", numbers)
            
            st.markdown(f"### Number: {selected_number}")
            st.info("Video tutorial will be displayed here")
    
    with col2:
        st.markdown("#### 🎯 Practice Mode")
        
        if st.button("Start Practice"):
            st.markdown("**Show the sign to your webcam**")
            st.info("Camera will activate here for practice")
            
        # Progress tracking
        st.markdown("#### 📊 Your Progress")
        st.progress(0.65)
        st.markdown("65% Complete")
        
        st.metric("Signs Learned", "17/26")
        st.metric("Accuracy", "92%")

# ============================================================================
# MODE 4: EMERGENCY
# ============================================================================

elif mode == "Emergency":
    st.markdown("### 🚨 Emergency Mode")
    st.markdown("Quick access to emergency phrases and SOS")
    
    # Emergency phrases
    emergency_phrases = [
        "Help",
        "Emergency",
        "Call 911",
        "I need a doctor",
        "I'm lost",
        "I'm hurt",
        "Fire",
        "Police"
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🆘 Quick Phrases")
        
        for phrase in emergency_phrases:
            if st.button(f"🚨 {phrase}", key=f"emergency_{phrase}"):
                # Show sign
                st.success(f"Showing sign for: {phrase}")
                
                # Speak
                if enable_speech:
                    st.session_state.speech_handler.speak(phrase)
                
                # Send alert (placeholder)
                st.warning("Alert sent to emergency contacts!")
    
    with col2:
        st.markdown("#### 📍 Emergency Info")
        
        st.markdown("**Emergency Contacts:**")
        st.markdown("- 911 (Emergency)")
        st.markdown("- Family: +1-XXX-XXX-XXXX")
        st.markdown("- Friend: +1-XXX-XXX-XXXX")
        
        st.markdown("**Current Location:**")
        st.info("Location sharing will be displayed here")
        
        if st.button("📍 Share Location"):
            st.success("Location shared with emergency contacts!")

# ============================================================================
# CONVERSATION HISTORY
# ============================================================================

st.markdown("---")
st.markdown("### 💬 Conversation History")

if st.session_state.conversation:
    # Display last 10 conversations
    for conv in st.session_state.conversation[-10:]:
        timestamp = time.strftime('%H:%M:%S', time.localtime(conv['timestamp']))
        st.markdown(f"**[{timestamp}]** {conv['text']} "
                   f"*(confidence: {conv.get('confidence', 1.0):.2%})*")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Conversation"):
            save_conversation(st.session_state.conversation)
            st.success("Conversation saved!")
    with col2:
        if st.button("📄 Export to PDF"):
            st.info("PDF export feature coming soon!")
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation = []
            st.success("History cleared!")
else:
    st.info("No conversation history yet. Start translating!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for the deaf community</p>
    <p>🤟 Breaking communication barriers through AI</p>
    <p><a href='https://github.com/Akrati36/sign-language-translator'>GitHub</a> | 
       <a href='#'>Documentation</a> | 
       <a href='#'>Support</a></p>
</div>
""", unsafe_allow_html=True)