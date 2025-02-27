import mediapipe as mp
import cv2
import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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
        tmp = tmp.values.tolist()
        predictions.append(model.predict([tmp])[0])
    
    if show:
        print(confusion_matrix(predictions, target), '\n')
        print(classification_report(predictions, target))
    
    return predictions

def real_time_detection(model, teacher_data=None):
    """
    Perform real-time pose detection using the webcam.
    """
    # Get teacher data if available
    reference_angles_dict = {}
    
    if teacher_data and os.path.exists(teacher_data):
        try:
            # Load the teacher data CSV
            teacher_df = pd.read_csv(teacher_data)
            
            # Print column names to understand the structure
            print("Teacher data columns:", teacher_df.columns.tolist())
            
            # Check if the first column contains pose names (common format)
            # If not, we'll just use the angle data without pose-specific feedback
            if len(teacher_df.columns) > 12:  # At least 12 angle columns + possible pose column
                for i, row in teacher_df.iterrows():
                    # Try to get pose name from the first column
                    pose_name = row.iloc[0]  # Assume first column has pose names
                    angles = row.iloc[1:13].values.tolist()  # Get the next 12 columns as angles
                    if isinstance(pose_name, str):
                        reference_angles_dict[pose_name] = angles
                        print(f"Loaded reference angles for pose: {pose_name}")
            
            # If we couldn't load pose-specific data, just note that
            if not reference_angles_dict:
                print("Could not identify pose-specific data in the teacher CSV.")
                print("Will continue without pose-specific feedback.")
        
        except Exception as e:
            print(f"Error loading teacher data: {e}")
            print("Will continue without pose-specific feedback.")
    
    # Start camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Set up display
    feedback_text = ""
    confidence_threshold = 0.6  # Minimum confidence to display pose name
    
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
                # Get prediction and probability
                pose_class = model.predict([list_angles])[0]
                pose_probs = model.predict_proba([list_angles])[0]
                pose_prob = np.max(pose_probs)
                
                # Display pose name and confidence
                feedback = f"Pose: {pose_class}"
                conf_text = f"Confidence: {pose_prob:.2f}"
                
                color = (0, 255, 0) if pose_prob > confidence_threshold else (0, 165, 255)
                
                cv2.putText(img_with_landmarks, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img_with_landmarks, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # If we have reference angles for the detected pose, provide feedback
                if pose_class in reference_angles_dict and pose_prob > confidence_threshold:
                    reference_angles = reference_angles_dict[pose_class]
                    if len(reference_angles) == len(list_angles):
                        angle_diff = [abs(a - b) for a, b in zip(list_angles, reference_angles)]
                        max_diff_idx = np.argmax(angle_diff)
                        
                        # Map index to body part
                        body_parts = ["Left Wrist", "Right Wrist", "Left Elbow", "Right Elbow", 
                                     "Left Shoulder", "Right Shoulder", "Left Knee", "Right Knee",
                                     "Left Ankle", "Right Ankle", "Left Hip", "Right Hip"]
                        
                        # Provide feedback based on largest angle difference
                        if angle_diff[max_diff_idx] > 20:  # If difference is significant
                            feedback_text = f"Adjust your {body_parts[max_diff_idx]}"
                            cv2.putText(img_with_landmarks, feedback_text, (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the image with landmarks
            cv2.imshow("Yoga Pose Recognition", img_with_landmarks)
            
        else:
            # If no pose detected, show original image
            cv2.putText(img, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Yoga Pose Recognition", img)
            
        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Main execution
def main():
    # Load the data and train the model
    try:
        # Try to find data in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, "train_angle.csv")
        test_path = os.path.join(script_dir, "test_angle.csv")
        
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        
        # Split features and target
        X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
        
        # Train SVM model
        model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
        model.fit(X, Y)
        
        print("Model trained successfully!")
        
        # Evaluate the model
        predictions = evaluate(data_test, model, show=True)
        
        # Create and display confusion matrix
        cm = confusion_matrix(data_test['target'], predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Start real-time yoga pose detection
        print("\nStarting real-time yoga pose detection...")
        print("Press 'q' to quit")
        
        # Look for teacher data
        teacher_data = os.path.join(script_dir, "teacher_yoga/angle_teacher_yoga.csv")
        if not os.path.exists(teacher_data):
            teacher_data = None
            print("Teacher reference data not found.")
        
        # Run real-time detection
        real_time_detection(model, teacher_data)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CSV files exist in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()