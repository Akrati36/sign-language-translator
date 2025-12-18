"""
Hand Detector - MediaPipe-based hand landmark detection
Detects hands in video frames and extracts 21 landmark points
"""

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    """Detect hands and extract landmarks using MediaPipe"""
    
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame):
        """
        Detect hands in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        # Extract landmarks
        hands_data = {
            'detected': False,
            'num_hands': 0,
            'landmarks': [],
            'handedness': []
        }
        
        if results.multi_hand_landmarks:
            hands_data['detected'] = True
            hands_data['num_hands'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks)
                hands_data['landmarks'].append(landmarks)
            
            # Get handedness (left/right)
            if results.multi_handedness:
                for hand_info in results.multi_handedness:
                    handedness = hand_info.classification[0].label
                    hands_data['handedness'].append(handedness)
        
        return hands_data
    
    def _extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            NumPy array of shape (21, 3) with x, y, z coordinates
        """
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])
        
        return np.array(landmarks)
    
    def draw_landmarks(self, frame, landmarks_list):
        """
        Draw hand landmarks on frame
        
        Args:
            frame: Input image
            landmarks_list: List of landmark arrays
            
        Returns:
            Frame with drawn landmarks
        """
        # Convert landmarks back to MediaPipe format
        for landmarks in landmarks_list:
            # Create landmark list
            landmark_list = self.mp_hands.HandLandmark
            
            # Draw connections
            h, w, c = frame.shape
            for i in range(21):
                x = int(landmarks[i][0] * w)
                y = int(landmarks[i][1] * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections between landmarks
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_x = int(landmarks[start_idx][0] * w)
                start_y = int(landmarks[start_idx][1] * h)
                end_x = int(landmarks[end_idx][0] * w)
                end_y = int(landmarks[end_idx][1] * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                        (255, 0, 0), 2)
        
        return frame
    
    def get_bounding_box(self, landmarks):
        """
        Get bounding box around hand
        
        Args:
            landmarks: Landmark array (21, 3)
            
        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        return (x_min, y_min, x_max, y_max)
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to be translation and scale invariant
        
        Args:
            landmarks: Landmark array (21, 3)
            
        Returns:
            Normalized landmarks
        """
        # Center around wrist (landmark 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        middle_finger_tip = landmarks[12]
        hand_size = np.linalg.norm(middle_finger_tip - wrist)
        
        if hand_size > 0:
            normalized = centered / hand_size
        else:
            normalized = centered
        
        return normalized
    
    def get_hand_orientation(self, landmarks):
        """
        Get hand orientation (palm facing camera or not)
        
        Args:
            landmarks: Landmark array (21, 3)
            
        Returns:
            String: 'front' or 'back'
        """
        # Use z-coordinates to determine orientation
        # If palm is facing camera, z values are more negative
        avg_z = np.mean(landmarks[:, 2])
        
        if avg_z < -0.05:
            return 'front'
        else:
            return 'back'
    
    def close(self):
        """Release resources"""
        self.hands.close()