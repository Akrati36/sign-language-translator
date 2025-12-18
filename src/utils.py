"""
Utility Functions
Helper functions for the application
"""

import json
import os
from datetime import datetime
import sqlite3

def load_config(config_path='config.json'):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'model_path': 'models/sign_classifier.h5',
            'confidence_threshold': 0.7,
            'max_hands': 2,
            'language': 'ASL',
            'enable_speech': True,
            'speech_rate': 150,
            'speech_volume': 1.0
        }

def save_conversation(conversation, db_path='database/conversations.db'):
    """
    Save conversation to database
    
    Args:
        conversation: List of conversation items
        db_path: Path to SQLite database
    """
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            type TEXT,
            text TEXT,
            confidence REAL,
            session_id TEXT
        )
    ''')
    
    # Generate session ID
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Insert conversation items
    for item in conversation:
        cursor.execute('''
            INSERT INTO conversations (timestamp, type, text, confidence, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            item.get('timestamp', datetime.now().timestamp()),
            item.get('type', 'sign'),
            item.get('text', ''),
            item.get('confidence', 1.0),
            session_id
        ))
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"✓ Saved conversation (session: {session_id})")

def load_conversations(db_path='database/conversations.db', limit=10):
    """
    Load recent conversations from database
    
    Args:
        db_path: Path to SQLite database
        limit: Number of recent conversations to load
        
    Returns:
        List of conversation items
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, type, text, confidence, session_id
        FROM conversations
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    conversations = []
    for row in rows:
        conversations.append({
            'timestamp': row[0],
            'type': row[1],
            'text': row[2],
            'confidence': row[3],
            'session_id': row[4]
        })
    
    return conversations

def export_to_pdf(conversation, output_path='exports/conversation.pdf'):
    """
    Export conversation to PDF
    
    Args:
        conversation: List of conversation items
        output_path: Output PDF path
    """
    # TODO: Implement PDF export using reportlab
    print(f"PDF export to {output_path} - Coming soon!")

def export_to_text(conversation, output_path='exports/conversation.txt'):
    """
    Export conversation to text file
    
    Args:
        conversation: List of conversation items
        output_path: Output text file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Sign Language Conversation\n")
        f.write("=" * 50 + "\n\n")
        
        for item in conversation:
            timestamp = datetime.fromtimestamp(item['timestamp'])
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] ")
            f.write(f"{item['text']} ")
            f.write(f"(confidence: {item.get('confidence', 1.0):.2%})\n")
    
    print(f"✓ Exported to {output_path}")

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate prediction accuracy
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        
    Returns:
        Accuracy score
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    accuracy = correct / len(predictions)
    
    return accuracy

def format_time(seconds):
    """
    Format seconds to readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"