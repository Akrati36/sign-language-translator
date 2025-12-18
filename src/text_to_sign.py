"""
Text to Sign - Convert text to sign language
Provides sign language animations/videos for text input
"""

import json
import os

class TextToSign:
    """Convert text to sign language"""
    
    def __init__(self, signs_database='data/signs_database.json'):
        """
        Initialize text to sign converter
        
        Args:
            signs_database: Path to signs database
        """
        self.signs_database = signs_database
        self.signs_dict = {}
        self.load_signs_database()
    
    def load_signs_database(self):
        """Load signs database"""
        if os.path.exists(self.signs_database):
            try:
                with open(self.signs_database, 'r') as f:
                    self.signs_dict = json.load(f)
                print(f"✓ Loaded {len(self.signs_dict)} signs from database")
            except Exception as e:
                print(f"⚠ Error loading signs database: {e}")
                self._create_default_database()
        else:
            print("⚠ Signs database not found, creating default")
            self._create_default_database()
    
    def _create_default_database(self):
        """Create default signs database"""
        # Default database with common words
        self.signs_dict = {
            # Alphabet
            **{letter: {
                'word': letter,
                'type': 'letter',
                'description': f'Sign for letter {letter}',
                'animation': f'animations/letters/{letter.lower()}.gif',
                'video': f'videos/letters/{letter.lower()}.mp4'
            } for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
            
            # Numbers
            **{str(num): {
                'word': str(num),
                'type': 'number',
                'description': f'Sign for number {num}',
                'animation': f'animations/numbers/{num}.gif',
                'video': f'videos/numbers/{num}.mp4'
            } for num in range(10)},
            
            # Common words
            'hello': {
                'word': 'hello',
                'type': 'word',
                'description': 'Wave hand near head',
                'animation': 'animations/words/hello.gif',
                'video': 'videos/words/hello.mp4'
            },
            'thank you': {
                'word': 'thank you',
                'type': 'phrase',
                'description': 'Touch chin and move hand forward',
                'animation': 'animations/phrases/thank_you.gif',
                'video': 'videos/phrases/thank_you.mp4'
            },
            'please': {
                'word': 'please',
                'type': 'word',
                'description': 'Rub chest in circular motion',
                'animation': 'animations/words/please.gif',
                'video': 'videos/words/please.mp4'
            },
            'sorry': {
                'word': 'sorry',
                'type': 'word',
                'description': 'Rub fist on chest in circular motion',
                'animation': 'animations/words/sorry.gif',
                'video': 'videos/words/sorry.mp4'
            },
            'help': {
                'word': 'help',
                'type': 'word',
                'description': 'Place fist on palm and lift up',
                'animation': 'animations/words/help.gif',
                'video': 'videos/words/help.mp4'
            },
            'yes': {
                'word': 'yes',
                'type': 'word',
                'description': 'Nod fist up and down',
                'animation': 'animations/words/yes.gif',
                'video': 'videos/words/yes.mp4'
            },
            'no': {
                'word': 'no',
                'type': 'word',
                'description': 'Snap index and middle finger to thumb',
                'animation': 'animations/words/no.gif',
                'video': 'videos/words/no.mp4'
            }
        }
    
    def translate(self, text):
        """
        Translate text to sign language
        
        Args:
            text: Input text
            
        Returns:
            List of sign dictionaries
        """
        # Clean and split text
        words = text.lower().strip().split()
        
        signs = []
        
        for word in words:
            # Check if word exists in database
            if word in self.signs_dict:
                signs.append(self.signs_dict[word])
            else:
                # Spell out word letter by letter
                for letter in word:
                    if letter.upper() in self.signs_dict:
                        signs.append(self.signs_dict[letter.upper()])
        
        return signs
    
    def get_sign_info(self, word):
        """
        Get sign information for a word
        
        Args:
            word: Word to look up
            
        Returns:
            Sign dictionary or None
        """
        word_lower = word.lower()
        word_upper = word.upper()
        
        if word_lower in self.signs_dict:
            return self.signs_dict[word_lower]
        elif word_upper in self.signs_dict:
            return self.signs_dict[word_upper]
        else:
            return None
    
    def add_sign(self, word, sign_info):
        """
        Add new sign to database
        
        Args:
            word: Word/phrase
            sign_info: Dictionary with sign information
        """
        self.signs_dict[word.lower()] = sign_info
        self._save_database()
    
    def _save_database(self):
        """Save signs database to file"""
        try:
            os.makedirs(os.path.dirname(self.signs_database), exist_ok=True)
            with open(self.signs_database, 'w') as f:
                json.dump(self.signs_dict, f, indent=2)
            print("✓ Signs database saved")
        except Exception as e:
            print(f"⚠ Error saving database: {e}")
    
    def get_all_signs(self):
        """Get all signs in database"""
        return self.signs_dict
    
    def search_signs(self, query):
        """
        Search for signs matching query
        
        Args:
            query: Search query
            
        Returns:
            List of matching signs
        """
        query_lower = query.lower()
        matches = []
        
        for word, sign_info in self.signs_dict.items():
            if query_lower in word.lower():
                matches.append(sign_info)
        
        return matches