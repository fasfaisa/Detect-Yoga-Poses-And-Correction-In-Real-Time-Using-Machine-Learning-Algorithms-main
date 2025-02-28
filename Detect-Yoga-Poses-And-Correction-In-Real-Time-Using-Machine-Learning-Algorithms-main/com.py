import mediapipe as mp
import cv2
import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras.models import load_model, model_from_json
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Emotion mapping to recommended yoga poses

EMOTION_TO_POSE = {
    "Angry": ["warrior2", "plank", "goddess"],  # Substituting with available poses
    "Disgusted": ["tree", "downdog", "plank"],  # Substituting with available poses
    "Fearful": ["plank", "tree", "goddess"],    # Substituting with available poses
    "Happy": ["warrior2", "tree", "downdog"],
    "Neutral": ["downdog", "warrior2", "tree"],
    "Sad": ["goddess", "warrior2", "downdog"],  # Substituting with available poses
    "Surprised": ["tree", "goddess", "plank"]   # Substituting with available poses
}


# Load emotion detection model - UPDATED to support both formats
def load_emotion_model(model_json_path="emotion_model.json", model_weights_path="emotion_model.h5", 
                       model_path="emotion_model.h5"):
    try:
        # First try loading from json + weights (as in your emotion training code)
        if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
            print(f"Loading model from JSON and weights...")
            with open(model_json_path, "r") as json_file:
                loaded_model_json = json_file.read()
                model = model_from_json(loaded_model_json)
            model.load_weights(model_weights_path)
            # Compile the model with the same settings used in training
            model.compile(loss='categorical_crossentropy', 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                          metrics=['accuracy'])
            print(f"Emotion model loaded successfully from JSON and weights")
            return model
        # Fallback to loading full model (as in your original code)
        elif os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Emotion model loaded successfully from {model_path}")
            return model
        else:
            print(f"Model files not found.")
            return None
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None

def preprocess_face_for_emotion(face_img):
    try:
        resized = cv2.resize(face_img, (48, 48))
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = resized / 255.0
        processed_face = np.expand_dims(np.expand_dims(normalized, -1), 0)
        
        # Debug: Print shape and content
        print(f"Processed face shape: {processed_face.shape}")
        print(f"Processed face content: {processed_face}")
        
        return processed_face
    except Exception as e:
        print(f"Error preprocessing face: {e}")
        return None
def detect_emotion(image, emotion_model):
    if emotion_model is None:
        return "neutral", 0.0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        print("No face detected")
        return "neutral", 0.0
    
    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box
    h, w, _ = image.shape
    x_min = int(box.xmin * w)
    y_min = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    width = min(width, w - x_min)
    height = min(height, h - y_min)
    
    face_img = image[y_min:y_min+height, x_min:x_min+width]
    
    if face_img.size == 0:
        print("Extracted face has zero size")
        return "neutral", 0.0
    
    # Visualize the extracted face
    cv2.imshow("Extracted Face", face_img)
    cv2.waitKey(1)  # Wait for 1 ms to update the window
    
    processed_face = preprocess_face_for_emotion(face_img)
    
    if processed_face is None:
        print("Face preprocessing failed")
        return "neutral", 0.0
    
    emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    try:
        emotion_probs = emotion_model.predict(processed_face)[0]
        emotion_idx = np.argmax(emotion_probs)
        
        # Print probabilities for debugging
        emotion_debug = {emotions[i]: float(emotion_probs[i]) for i in range(len(emotions))}
        print(f"Emotion probabilities: {emotion_debug}")
        
        return emotions[emotion_idx], emotion_probs[emotion_idx]
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return "neutral", 0.0

def calculate_angle(landmark1, landmark2, landmark3):
    """Calculate the angle between three landmarks."""
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    return angle

def extract_pose_angles(results):
    """Extract angles from pose landmarks."""
    angles = []
        
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        
        # Left wrist angle
        left_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])
        angles.append(left_wrist_angle)
        
        # Right wrist angle
        right_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])
        angles.append(right_wrist_angle)

        # Left elbow angle
        left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        angles.append(left_elbow_angle)
        
        # Right elbow angle
        right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        angles.append(right_elbow_angle)
        
        # Left shoulder angle
        left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        angles.append(left_shoulder_angle)

        # Right shoulder angle
        right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        angles.append(right_shoulder_angle)

        # Left knee angle
        left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        angles.append(left_knee_angle)

        # Right knee angle
        right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        angles.append(right_knee_angle)

        # Left ankle angle
        left_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
        angles.append(left_ankle_angle)

        # Right ankle angle
        right_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
        angles.append(right_ankle_angle)

        # Left hip angle
        left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        angles.append(left_hip_angle)

        # Right hip angle
        right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        angles.append(right_hip_angle)
    
    return angles
def evaluate(data_test, model, show=False):
    """Evaluate model on test data."""
    target = data_test.loc[:, "target"]  # list of labels
    target = target.values.tolist()
    predictions = []
    for i in range(len(data_test)):
        tmp = data_test.iloc[i, 0:len(data_test.columns) - 1]
        tmp = tmp.values.reshape(1, -1)  # Reshape to 2D array
        predictions.append(model.predict(tmp)[0])
    
    if show:
        print(confusion_matrix(predictions, target), '\n')
        print(classification_report(predictions, target))
    
    return predictions
def load_reference_angles(teacher_data):
    """
    Load reference angles from the teacher data CSV file.
    """
    reference_angles_dict = {}
    
    if teacher_data and os.path.exists(teacher_data):
        try:
            # Load the teacher data CSV
            teacher_df = pd.read_csv(teacher_data)
            
            # Print column names to understand the structure
            print("Teacher data columns:", teacher_df.columns.tolist())
            
            # Look for pose-specific data
            for i, row in teacher_df.iterrows():
                try:
                    # Assume last column has pose names
                    angles = row.iloc[:12].values.tolist()  # Get the first 12 columns as angles
                    pose_name = row.iloc[12]  # Pose name in column 12 (Fixed the typo 'ilc' -> 'iloc')
                    if isinstance(pose_name, str):
                        reference_angles_dict[pose_name] = angles
                        print(f"Loaded reference angles for pose: {pose_name}")
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue
        except Exception as e:
            print(f"Error loading teacher data: {e}")
    
    return reference_angles_dict
def real_time_detection(pose_model, teacher_data=None):
    """
    Perform real-time pose detection using the webcam with enhanced feedback.
    """
    # Load teacher data if available
    reference_angles_dict = load_reference_angles(teacher_data)
    
    # Body part names for feedback
    angle_name_list = ["Left Wrist", "Right Wrist", "Left Elbow", "Right Elbow", 
                      "Left Shoulder", "Right Shoulder", "Left Knee", "Right Knee",
                      "Left Ankle", "Right Ankle", "Left Hip", "Right Hip"]

    # Define angle coordinates for detailed feedback
    angle_coordinates = [
        [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.LEFT_INDEX.value],
        [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.RIGHT_INDEX.value],
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
        [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
        [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]
    ]

    # Start camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Set up display parameters
    confidence_threshold = 0.6  # Minimum confidence to display pose name
    correction_value = 30  # Acceptable angle difference threshold
    fps_time = time.time()
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    print("Starting real-time pose detection. Press 'q' to quit.")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Convert to RGB and process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks
            img_with_landmarks = img.copy()
            mp_drawing.draw_landmarks(img_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract angles and predict pose
            list_angles = extract_pose_angles(results)
            
            if len(list_angles) == 12:  # Make sure we have all angles
                # Reshape angles to 2D array for prediction
                angles_array = np.array(list_angles).reshape(1, -1)
                
                # Get prediction and probability
                pose_class = pose_model.predict(angles_array)[0]
                pose_probs = pose_model.predict_proba(angles_array)[0]
                pose_prob = np.max(pose_probs)
                
                # Display pose name and confidence
                feedback = f"Pose: {pose_class}"
                conf_text = f"Confidence: {pose_prob:.2f}"
                
                color = (0, 255, 0) if pose_prob > confidence_threshold else (0, 165, 255)
                
                cv2.putText(img_with_landmarks, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img_with_landmarks, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Enhanced feedback if we have reference angles for the detected pose
                if pose_class in reference_angles_dict and pose_prob > confidence_threshold:
                    reference_angles = reference_angles_dict[pose_class]
                    
                    # Count correct angles
                    correct_angle_count = 0
                    
                    # Check each angle for feedback
                    for i in range(12):
                        if i < len(reference_angles) and i < len(list_angles):
                            # Get angle difference
                            angle_diff = abs(list_angles[i] - reference_angles[i])
                            
                            # Determine status
                            if list_angles[i] < reference_angles[i] - correction_value:
                                status = "more"
                            elif reference_angles[i] + correction_value < list_angles[i]:
                                status = "less"
                            else:
                                status = "OK"
                                correct_angle_count += 1
                            
                            # Get landmarks for this angle
                            landmarks = results.pose_landmarks.landmark
                            idx1, idx2, idx3 = angle_coordinates[i]
                            
                            # Calculate position to display status near the joint
                            point_b = (int(landmarks[idx2].x * img.shape[1]), 
                                      int(landmarks[idx2].y * img.shape[0]))
                            
                            # Display status near joint
                            status_position = (point_b[0] - int(img.shape[1] * 0.03), 
                                             point_b[1] + int(img.shape[0] * 0.03))
                            
                            status_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
                            cv2.putText(img_with_landmarks, status, status_position, 
                                       cv2.FONT_HERSHEY_PLAIN, 1, status_color, 2)
                            
                            # Display angle names
                            cv2.putText(img_with_landmarks, angle_name_list[i], 
                                       (point_b[0] - 50, point_b[1] - 10), 
                                       cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
                    
                    # Overall posture feedback
                    posture = "CORRECT" if correct_angle_count >= 9 else "WRONG"
                    posture_color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)
                    cv2.putText(img_with_landmarks, f"Yoga movements: {posture}", (20, 100), 
                               cv2.FONT_HERSHEY_PLAIN, 1.5, posture_color, 2)
                
                # Display FPS
                fps = 1.0 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(img_with_landmarks, f"FPS: {fps:.1f}", (20, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
                # Display the image with landmarks
                cv2.imshow("Yoga Pose Recognition", img_with_landmarks)
            else:
                # If angles couldn't be extracted properly
                cv2.putText(img, "Pose landmarks incomplete", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Yoga Pose Recognition", img)
        else:
            # If no pose detected, show original image
            cv2.putText(img, "No pose detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Yoga Pose Recognition", img)
            
        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def combined_realtime_emotion_yoga(pose_model, emotion_model, teacher_data=None):
    """
    Combined real-time emotion detection and yoga pose recognition with enhanced feedback.
    """
    # Load teacher data if available
    reference_angles_dict = load_reference_angles(teacher_data)
    
    # Debug: Print loaded reference angles
    if reference_angles_dict:
        print("Loaded reference angles for poses:")
        for pose_name, angles in reference_angles_dict.items():
            print(f"{pose_name}: {angles}")
    else:
        print("No reference angles loaded. Continuing without pose-specific feedback.")
    
    # Create MediaPipe pose object
    mp_pose_processor = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Body part names for feedback
    angle_name_list = ["Left Wrist", "Right Wrist", "Left Elbow", "Right Elbow", 
                      "Left Shoulder", "Right Shoulder", "Left Knee", "Right Knee",
                      "Left Ankle", "Right Ankle", "Left Hip", "Right Hip"]
    
    # Create angle_coordinates for detailed feedback
    angle_coordinates = [
        [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.LEFT_INDEX.value],
        [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.RIGHT_INDEX.value],
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
        [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
        [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
        [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
        [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]
    ]
    
    # Start camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Set up display parameters
    confidence_threshold = 0.6  # Minimum confidence to display pose name
    correction_value = 30  # Acceptable angle difference threshold
    fps_time = time.time()
    font_scale = 0.7
    line_height = 30
    
    # Variables to track emotion state
    current_emotion = "neutral"
    emotion_stability_counter = 0
    emotion_confirmed = False
    recommended_poses = []
    selected_pose = None
    mode = "emotion_detection"  # Start in emotion detection mode
    
    emotion_display_time = 0
    recommendation_display_time = 0
    
    # Debug counter
    frame_counter = 0
    
    print("Starting combined emotion-based yoga assistant. Press 'q' to quit, 'r' to reset to emotion detection.")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Convert to RGB and process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a copy for overlay
        overlay = img.copy()
        
        # Debug info - every 30 frames
        frame_counter += 1
        if frame_counter % 30 == 0:
            print(f"Current mode: {mode}, Current emotion: {current_emotion}")
        
        if mode == "emotion_detection":
            # Detect emotion
            emotion, emotion_prob = detect_emotion(img, emotion_model)
            
            # Debug - print the detected emotion and probability
            if frame_counter % 30 == 0:
                print(f"Detected emotion: {emotion}, probability: {emotion_prob:.2f}")
            
            # Draw emotion text with probability
            cv2.putText(overlay, f"Detected Emotion: {emotion.upper()}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            cv2.putText(overlay, f"Confidence: {emotion_prob:.2f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            
            # Keep track of stable emotion
            if emotion == current_emotion:
                emotion_stability_counter += 1
            else:
                current_emotion = emotion
                emotion_stability_counter = 0
            
            # If emotion is stable for 30 frames, recommend poses
            if emotion_stability_counter >= 30 and not emotion_confirmed:
                emotion_confirmed = True
                recommended_poses = EMOTION_TO_POSE.get(emotion, ["Mountain Pose"])
                mode = "pose_recommendation"
                recommendation_display_time = 0
                print(f"Emotion {emotion} confirmed. Recommending poses: {recommended_poses}")
        
        elif mode == "pose_recommendation":
            # Display recommended poses
            recommendation_display_time += 1
            
            cv2.putText(overlay, f"For your {current_emotion.upper()} emotion, we recommend:", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            
            for i, pose in enumerate(recommended_poses):
                cv2.putText(overlay, f"{i+1}. {pose}", (40, 70 + i*line_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            
            cv2.putText(overlay, "Press 1, 2, or 3 to select a pose, or 'r' to reset", 
                        (20, 70 + len(recommended_poses)*line_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            
            # If displayed for 10 seconds, move to pose detection mode
            if recommendation_display_time > 300:  # ~10 seconds at 30fps
                mode = "pose_detection"
                selected_pose = recommended_poses[0]  # Default to first pose
                print(f"Selected pose: {selected_pose}")
        
        elif mode == "pose_detection":
            # Process pose with MediaPipe
            pose_results = mp_pose_processor.process(img_rgb)
            
            if pose_results.pose_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(overlay, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract angles and predict pose
                list_angles = extract_pose_angles(pose_results)
                
                if len(list_angles) == 12:  # Make sure we have all angles
                    # Get prediction and probability
                    pose_class = pose_model.predict([list_angles])[0]
                    pose_probs = pose_model.predict_proba([list_angles])[0]
                    pose_prob = np.max(pose_probs)
                    
                    # Display pose name and confidence
                    feedback = f"Detected Pose: {pose_class}"
                    conf_text = f"Confidence: {pose_prob:.2f}"
                    target_text = f"Target Pose: {selected_pose}"
                    
                    color = (0, 255, 0) if pose_prob > confidence_threshold else (0, 165, 255)
                    
                    cv2.putText(overlay, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    cv2.putText(overlay, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    cv2.putText(overlay, target_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                    
                    # Display FPS
                    fps = 1.0 / (time.time() - fps_time)
                    fps_time = time.time()
                    cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    
                    # Provide feedback based on detected pose
                    if pose_class.lower() == selected_pose.lower() and pose_prob > confidence_threshold:
                        cv2.putText(overlay, "Great job! You're doing the correct pose!", (20, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    else:
                        cv2.putText(overlay, f"Try to get into the {selected_pose} pose", (20, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                    

                    # Enhanced feedback if we have reference angles for the selected pose
                    if selected_pose in reference_angles_dict and pose_prob > confidence_threshold:
                        reference_angles = reference_angles_dict[selected_pose]
                        
                        # Count correct angles
                        correct_angle_count = 0
                        
                        # Check each angle for feedback
                        for i in range(12):
                            if i < len(reference_angles) and i < len(list_angles):
                                # Get angle difference
                                angle_diff = abs(list_angles[i] - reference_angles[i])
                                
                                # Determine status
                                if list_angles[i] < reference_angles[i] - correction_value:
                                    status = "more"
                                elif reference_angles[i] + correction_value < list_angles[i]:
                                    status = "less"
                                else:
                                    status = "OK"
                                    correct_angle_count += 1
                                
                                # Get landmarks for this angle
                                landmarks = pose_results.pose_landmarks.landmark
                                idx1, idx2, idx3 = angle_coordinates[i]
                                
                                # Calculate position to display status near the joint
                                point_b = (int(landmarks[idx2].x * img.shape[1]), 
                                          int(landmarks[idx2].y * img.shape[0]))
                                
                                # Display status near joint
                                status_position = (point_b[0] - int(img.shape[1] * 0.03), 
                                                 point_b[1] + int(img.shape[0] * 0.03))
                                
                                status_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
                                cv2.putText(overlay, status, status_position, 
                                           cv2.FONT_HERSHEY_PLAIN, 1, status_color, 2)
                                
                                # Display angle names
                                cv2.putText(overlay, angle_name_list[i], 
                                           (point_b[0] - 50, point_b[1] - 10), 
                                           cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
                        
                        # Overall posture feedback
                        posture = "CORRECT" if correct_angle_count >= 9 else "WRONG"
                        posture_color = (0, 255, 0) if posture == "CORRECT" else (0, 0, 255)
                        cv2.putText(overlay, f"Yoga movements: {posture}", (20, 190), 
                                   cv2.FONT_HERSHEY_PLAIN, 1.5, posture_color, 2)
            
            else:
                # If no pose detected
                cv2.putText(overlay, "No pose detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                if selected_pose:
                    cv2.putText(overlay, f"Try to get into the {selected_pose} pose", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        # Show the image with overlay
        cv2.imshow("Emotion-Based Yoga Assistant", overlay)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset to emotion detection mode
            mode = "emotion_detection"
            emotion_confirmed = False
            emotion_stability_counter = 0
            selected_pose = None
        elif key == ord('1') and mode == "pose_recommendation" and len(recommended_poses) >= 1:
            mode = "pose_detection"
            selected_pose = recommended_poses[0]
            print(f"Selected pose: {selected_pose}")
        elif key == ord('2') and mode == "pose_recommendation" and len(recommended_poses) >= 2:
            mode = "pose_detection"
            selected_pose = recommended_poses[1]
            print(f"Selected pose: {selected_pose}")
        elif key == ord('3') and mode == "pose_recommendation" and len(recommended_poses) >= 3:
            mode = "pose_detection"
            selected_pose = recommended_poses[2]
            print(f"Selected pose: {selected_pose}")
            
    cap.release()
    cv2.destroyAllWindows()

# Update the main function to use the combined function
def main():
    # Load the data and train the pose model
    try:
        # Try to find data in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = os.getcwd()
            
        train_path = os.path.join(script_dir, "train_angle.csv")
        test_path = os.path.join(script_dir, "test_angle.csv")
        
        # Look for emotion model in both formats
        emotion_model_json = os.path.join(script_dir, "emotion_model.json")
        emotion_model_weights = os.path.join(script_dir, "emotion_model.h5")
        emotion_model_full = os.path.join(script_dir, "emotion_model.h5")
        
        # Check if training data files exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Training data not found at {train_path} or {test_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Script directory: {script_dir}")
            print("Please make sure the CSV files exist in the correct location.")
            return
            
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        
        # Split features and target
        X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
        
        # Train SVM model for pose detection
        pose_model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
        pose_model.fit(X, Y)
        
        print("Pose model trained successfully!")
        
        # Evaluate the model
        predictions = evaluate(data_test, pose_model, show=True)
        
        # Create and display confusion matrix
        cm = confusion_matrix(data_test['target'], predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=pose_model.classes_, yticklabels=pose_model.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Load emotion detection model with updated function
        emotion_model = load_emotion_model(
            model_json_path=emotion_model_json,
            model_weights_path=emotion_model_weights,
            model_path=emotion_model_full
        )
        
        # Look for teacher data in various possible locations
        teacher_data_paths = [
            os.path.join(script_dir, "angle_teacher_yoga.csv"),
            os.path.join(script_dir, "teacher_yoga/angle_teacher_yoga.csv"),
            os.path.join(script_dir, "../teacher_yoga/angle_teacher_yoga.csv"),
            os.path.join(script_dir, "Detect-Yoga-Poses-And-Correction-In-Real-Time-Using-Machine-Learning-Algorithms-main/teacher_yoga/angle_teacher_yoga.csv")
        ]
        
        teacher_data = None
        for path in teacher_data_paths:
            if os.path.exists(path):
                teacher_data = path
                print(f"Found teacher reference data at: {path}")
                break
                
        if not teacher_data:
            print("Teacher reference data not found. Continuing without feedback.")
        
        # Ask user which mode to run
        print("\nChoose a mode to run:")
        print("1. Basic pose detection")
        print("2. Emotion-based yoga assistant")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == "1":
            # Run basic pose detection
            print("\nStarting basic pose detection...")
            real_time_detection(pose_model, teacher_data)
        else:
            # Run emotion-based yoga assistant
            print("\nStarting emotion-based yoga assistant...")
            combined_realtime_emotion_yoga(pose_model, emotion_model, teacher_data)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CSV files exist in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()