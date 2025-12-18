"""
Speech Handler - Text-to-Speech and Speech-to-Text
Handles voice input and output for the application
"""

import pyttsx3
import speech_recognition as sr

class SpeechHandler:
    """Handle speech recognition and text-to-speech"""
    
    def __init__(self):
        """Initialize speech handler"""
        # Text-to-Speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            self.tts_available = True
        except Exception as e:
            print(f"⚠ TTS not available: {e}")
            self.tts_available = False
        
        # Speech recognition
        try:
            self.recognizer = sr.Recognizer()
            self.stt_available = True
        except Exception as e:
            print(f"⚠ STT not available: {e}")
            self.stt_available = False
    
    def _configure_tts(self):
        """Configure text-to-speech settings"""
        # Set properties
        self.tts_engine.setProperty('rate', 150)  # Speed
        self.tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        
        # Set voice (prefer female voice if available)
        for voice in voices:
            if 'female' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
        """
        if not self.tts_available:
            print(f"TTS not available. Would speak: {text}")
            return
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error speaking: {e}")
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """
        Listen to microphone and convert speech to text
        
        Args:
            timeout: Seconds to wait for speech
            phrase_time_limit: Maximum seconds for phrase
            
        Returns:
            Recognized text or None
        """
        if not self.stt_available:
            print("STT not available")
            return None
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                print("Processing...")
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                return text
                
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results: {e}")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
    
    def set_voice_properties(self, rate=None, volume=None, voice_id=None):
        """
        Set voice properties
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            voice_id: Voice ID to use
        """
        if not self.tts_available:
            return
        
        if rate is not None:
            self.tts_engine.setProperty('rate', rate)
        
        if volume is not None:
            self.tts_engine.setProperty('volume', volume)
        
        if voice_id is not None:
            self.tts_engine.setProperty('voice', voice_id)
    
    def get_available_voices(self):
        """
        Get list of available voices
        
        Returns:
            List of voice objects
        """
        if not self.tts_available:
            return []
        
        return self.tts_engine.getProperty('voices')
    
    def save_to_file(self, text, filename):
        """
        Save speech to audio file
        
        Args:
            text: Text to convert
            filename: Output filename
        """
        if not self.tts_available:
            return
        
        try:
            self.tts_engine.save_to_file(text, filename)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error saving to file: {e}")