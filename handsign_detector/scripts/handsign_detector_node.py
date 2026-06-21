#!/usr/bin/env python3
# ~/catkin_handsigns/src/handsign_detector/scripts/handsign_detector_node.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from collections import deque, Counter
import glob
import rospkg

class HandSignDetector:
    def __init__(self):
        rospy.init_node('handsign_detector_node')
        
        self.bridge = CvBridge()
        
        # MediaPipe
        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Config
        self.SEQUENCE_LENGTH = 30
        self.FEATURES_PER_FRAME = 258
        self.CONF_THRESH = 0.70
        self.IDLE_LABEL = "neutral"
        self.MIN_MOTION = 0.003
        self.NO_HAND_THRESH = 10
        
        # Get base directory
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('handsign_detector')
        self.base_dir = os.path.join(pkg_path, 'model')
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.sign_names = {}
        self.available_models = []
        
        # Current active model
        self.current_model_name = ""
        self.model = None
        self.scaler = None
        self.sign_names_list = []
        
        # Discover models from folder names
        self._discover_models()

        # Buffers
        self.buf = deque(maxlen=self.SEQUENCE_LENGTH)
        self.history = deque(maxlen=12)
        self.no_hand_counter = 0
        
        if self.available_models:
            self._switch_model(self.available_models[0])
        else:
            rospy.logerr(f"No models found in {self.base_dir}")
        
        # Publishers
        self.result_pub = rospy.Publisher('/handsign/detection', String, queue_size=5)
        self.viz_pub = rospy.Publisher('/handsign/viz', Image, queue_size=2)
        
        # Subscriber for model switching
        self.model_switch_sub = rospy.Subscriber('/handsign/switch_model', String, 
                                                  self.switch_model_callback, queue_size=2)
        
        # Subscriber for camera
        self.image_sub = rospy.Subscriber('/camera/rgb', Image, self.image_callback, queue_size=2)
        
        rospy.loginfo(f"HandSign Detector ready")
        rospy.loginfo(f"Available models: {self.available_models}")
        rospy.loginfo(f"Current model: {self.current_model_name}")
    
    def _discover_models(self):
        """Discover models from folder names inside base_dir"""
        
        if not os.path.exists(self.base_dir):
            rospy.logerr(f"Directory not found: {self.base_dir}")
            return
        
        # Get all subdirectories (each one is a model)
        for folder_name in sorted(os.listdir(self.base_dir)):
            folder_path = os.path.join(self.base_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Find model file
            model_file = None
            for ext in ['*.keras', '*.h5']:
                files = glob.glob(os.path.join(folder_path, ext))
                if files:
                    model_file = files[0]
                    break
            
            if not model_file:
                rospy.logwarn(f"No model file in {folder_name}")
                continue
            
            # Find scaler
            scaler_file = os.path.join(folder_path, "scaler.pkl")
            if not os.path.exists(scaler_file):
                rospy.logwarn(f"No scaler.pkl in {folder_name}")
                continue
            
            # Find label encoder
            label_file = os.path.join(folder_path, "label_encoder.pkl")
            if not os.path.exists(label_file):
                rospy.logwarn(f"No label_encoder.pkl in {folder_name}")
                continue
            
            # Load model
            try:
                model = tf.keras.models.load_model(model_file)
                
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                
                with open(label_file, 'rb') as f:
                    le = pickle.load(f)
                
                # Store everything using folder name as key
                self.models[folder_name] = model
                self.scalers[folder_name] = scaler
                self.sign_names[folder_name] = list(le.classes_)
                self.available_models.append(folder_name)
                
                rospy.loginfo(f"  Loaded: {folder_name} -> {self.sign_names[folder_name]}")
                
            except Exception as e:
                rospy.logerr(f"Failed to load {folder_name}: {e}")
        
        rospy.loginfo(f"Loaded {len(self.available_models)} models")
    
    def _switch_model(self, model_name):
        """Switch to a different model"""
        if model_name not in self.models:
            rospy.logerr(f"Model '{model_name}' not found. Available: {self.available_models}")
            return False
        
        self.current_model_name = model_name
        self.model = self.models[model_name]
        self.scaler = self.scalers[model_name]
        self.sign_names_list = self.sign_names[model_name]
        
        # Clear buffers when switching
        self.buf.clear()
        self.history.clear()
        self.no_hand_counter = 0
        
        rospy.loginfo(f"Switched to: {model_name} | Signs: {self.sign_names_list}")
        return True
    
    def switch_model_callback(self, msg):
        """Callback for model switching via ROS topic"""
        model_name = msg.data.strip()
        
        if not model_name:
            rospy.logwarn("Empty model name received")
            return
        
        if model_name == "list":
            rospy.loginfo(f"Available models: {self.available_models}")
            rospy.loginfo(f"Current: {self.current_model_name}")
            return
        
        self._switch_model(model_name)
    
    def extract_all_landmarks(self, results):
        """Exactly matches original extract_all_landmarks()"""
        lm = []
        if results.pose_landmarks:
            for p in results.pose_landmarks.landmark:
                lm.extend([p.x, p.y, p.z, p.visibility])
        else:
            lm.extend([0] * 132)
        if results.left_hand_landmarks:
            for p in results.left_hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0] * 63)
        if results.right_hand_landmarks:
            for p in results.right_hand_landmarks.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0] * 63)
        return np.array(lm, dtype=np.float32)
    
    def image_callback(self, ros_image):
        if self.model is None:
            return
        
        frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        
        lh = results.left_hand_landmarks is not None
        rh = results.right_hand_landmarks is not None
        hands_present = lh or rh
        
        if not hands_present:
            self.no_hand_counter += 1
            if self.no_hand_counter > self.NO_HAND_THRESH:
                self.buf.clear()
                self.history.clear()
                self.no_hand_counter = 0
        else:
            self.no_hand_counter = 0
            self.buf.append(self.extract_all_landmarks(results))
            if len(self.buf) == self.SEQUENCE_LENGTH:           # only make prediction if buffer == 30
                seq = np.array(self.buf)
                motion = float(seq.std(axis=0).mean())
                if motion < self.MIN_MOTION:
                    if self.history:
                        self.history.popleft()
                else:
                    try:
                        seq_s = self.scaler.transform(seq.reshape(-1, self.FEATURES_PER_FRAME))
                        seq_s = seq_s.reshape(1, self.SEQUENCE_LENGTH, self.FEATURES_PER_FRAME).astype(np.float32)
                        pred = self.model.predict(seq_s, verbose=0)[0]
                        i = int(pred.argmax())
                        self.history.append((self.sign_names_list[i], float(pred[i])))
                    except Exception as e:
                        rospy.logerr_throttle(5, f"Prediction error: {e}")
        
        # Smoothed display via majority vote
        display, conf, is_sign = "...", 0.0, False
        if self.history:
            names = [p[0] for p in self.history]
            top = Counter(names).most_common(1)[0][0]
            conf = float(np.mean([p[1] for p in self.history if p[0] == top]))
            if top == self.IDLE_LABEL:
                display = "idle"
            elif conf >= self.CONF_THRESH:
                display, is_sign = top, True
            else:
                display = "uncertain"
        elif not hands_present:
            display = "(no hands)"
        
        # Publish result
        result_msg = String()
        result_msg.data = f"{display}|{conf:.2f}|{is_sign}|{self.current_model_name}"
        self.result_pub.publish(result_msg)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        if lh:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if rh:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        
        # Overlays
        col = (0, 255, 0) if is_sign else (0, 165, 255)
        cv2.putText(frame, display.upper(), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2)
        if conf:
            cv2.putText(frame, f"{conf:.0%}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        
        # Model name at top
        cv2.putText(frame, f"[{self.current_model_name}]", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        
        # Bottom status
        cv2.putText(frame,
                    f"L:{'Y' if lh else '-'} R:{'Y' if rh else '-'}  buf {len(self.buf)}/{self.SEQUENCE_LENGTH}",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Publish visualization
        viz_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.viz_pub.publish(viz_msg)
    
    def __del__(self):
        self.holistic.close()

if __name__ == '__main__':
    try:
        detector = HandSignDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass