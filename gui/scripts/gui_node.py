#!/usr/bin/env python3
"""
ROS Node for Hand Sign GUI.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time

from gui_app import SimpleHandSignGUI


class GUINode:
    def __init__(self):
        rospy.init_node('handsign_gui', anonymous=True)
        
        self.bridge = CvBridge()
        self.viz_sub = None
        self.result_sub = None
        self.voice_sub = None
        
        self.model_pub = rospy.Publisher('/handsign/switch_model', String, queue_size=2)
        self.sign_pub = rospy.Publisher('/handsign/target_sign', String, queue_size=2)
        self.tts_pub = rospy.Publisher('/handsign/speak', String, queue_size=2)
        self.llm_pub = rospy.Publisher('/handsign/llm_text_input', String, queue_size=2)
        
        self.expected_sign = ""
        self.correct_received = False
        self.last_time = 0

        self.nlp_signs = []
        self.nlp_models = []
        self.nlp_current_index = 0
        self.is_nlp_practice = False
        
        self.gui = SimpleHandSignGUI()
        self.gui.on_start = self.on_start
        self.gui.on_stop = self.on_stop
        self.gui.on_sign_change = self.on_sign_change
        self.gui.on_llm_text = self.send_llm_text
        
        # Subscribe to voice commands
        self.voice_sub = rospy.Subscriber('/handsign/voice_command', String, self.voice_cb)
        self.llm_sequence_sub = rospy.Subscriber('/handsign/llm_sequence', String, 
                                          self.llm_sequence_cb, queue_size=2)
        
        rospy.loginfo("GUI Node ready")

        rospy.sleep(3)
        self.tts_pub.publish(String(data="Welcome to Hand Sign Recognition Robot"))
    
    def on_start(self, module_folder, model_name):
        self.model_pub.publish(String(data=model_name))
        self._subscribe()
    
    def on_stop(self):
        self._unsubscribe()
    
    def on_sign_change(self, sign_name):
        self.expected_sign = sign_name.lower().strip()
        self.correct_received = False
        self.sign_pub.publish(String(data=sign_name))
        self.tts_pub.publish(String(data=f"Do this sign: {sign_name}"))
    
    def voice_cb(self, msg):
        """Handle voice commands."""
        text = msg.data.lower().strip()
        rospy.loginfo(f"Voice command: {text}")
        
        if "next" in text or "skip" in text:
            self.gui.next_class()
        elif "quit" in text or "stop" in text or "exit" in text:
            self.gui.stop_practice()
        elif "back" in text or "menu" in text:
            self.gui.stop_practice()
    
    def _subscribe(self):
        if self.viz_sub is None:
            self.viz_sub = rospy.Subscriber('/handsign/viz', Image, self.viz_cb, queue_size=2)
        if self.result_sub is None:
            self.result_sub = rospy.Subscriber('/handsign/detection', String, self.result_cb, queue_size=5)
    
    def _unsubscribe(self):
        if self.viz_sub:
            self.viz_sub.unregister()
            self.viz_sub = None
        if self.result_sub:
            self.result_sub.unregister()
            self.result_sub = None
    
    def viz_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.gui.update_frame(frame)
        except:
            pass
    
    def result_cb(self, msg):
        try:
            data = msg.data.split('|')
            if len(data) >= 3:
                detected = data[0].lower().strip()
                conf = float(data[1])
                is_sign = data[2] == 'True'
                
                if not is_sign:
                    return
                
                if time.time() - self.last_time < 0.8:
                    return
                self.last_time = time.time()
                
                # === NLP PRACTICE MODE ===
                if self.is_nlp_practice:
                    expected = self.nlp_signs[self.nlp_current_index]
                    
                    if detected == expected:
                        rospy.loginfo(f"✓ Correct: {detected}")
                        self.gui.update_nlp_progress(self.nlp_current_index, True)
                        self.nlp_current_index += 1
                        rospy.sleep(0.3)
                        self.load_next_nlp_sign()
                    elif conf > 0.5:
                        self.gui.update_nlp_progress(self.nlp_current_index, False)
                    return
                # === END NLP ===
                
                # Normal practice mode
                if self.correct_received:
                    return
                
                is_correct = (detected == self.expected_sign)
                
                if is_correct:
                    self.correct_received = True
                    self.gui.update_result(detected, conf, True)
                    self.tts_pub.publish(String(data="Correct"))
                elif conf > 0.5:
                    self.gui.update_result(detected, conf, False)
        except:
            pass
    
    def run(self):
        self.gui.run()
    
    def shutdown(self):
        self._unsubscribe()
        self.gui.quit()

    def send_llm_text(self, text):
        self.llm_pub.publish(String(data=text))

    def llm_sequence_cb(self, msg):
        """Receive sequence from LLM node"""
        
        if msg.data.startswith("ERROR"):
            if hasattr(self.gui, 'nlp_status'):
                self.gui.nlp_status.config(text="Failed! Limit your request to available handsigns.", fg='#f44')
            return
        
        sequence = msg.data.split('|')
        
        if not sequence:
            rospy.logwarn("No valid LLM sequence")
            return
        
        rospy.loginfo(f"NLP models: {sequence}")
        
        # Break each model into individual signs
        from gui_app import get_model_classes
        all_signs = []
        model_for_sign = []
        
        for model_path in sequence:
            module, model = model_path.split('/', 1)
            classes = get_model_classes(module, model)
            for sign in classes:
                all_signs.append(sign)
                model_for_sign.append((module, model))
        
        rospy.loginfo(f"Individual signs: {all_signs}")
        
        self.nlp_signs = all_signs
        self.nlp_models = model_for_sign
        self.nlp_current_index = 0
        self.is_nlp_practice = True
        
        self._subscribe()
        
        if hasattr(self.gui, 'show_nlp_practice_session'):
            self.gui.show_nlp_practice_session(all_signs)
        
        self.load_next_nlp_sign()

    def load_next_nlp_sign(self):
        """Load next individual sign in NLP sequence"""
        if self.nlp_current_index < len(self.nlp_signs):
            sign = self.nlp_signs[self.nlp_current_index]
            module, model = self.nlp_models[self.nlp_current_index]
            
            self.model_pub.publish(String(data=model))
            self.expected_sign = sign
            self.correct_received = False
            
            rospy.loginfo(f"NLP Sign {self.nlp_current_index+1}/{len(self.nlp_signs)}: {sign}")
            self.tts_pub.publish(String(data=f"Sign: {sign}"))
        else:
            self.tts_pub.publish(String(data="Practice complete!"))
            self.is_nlp_practice = False
            if hasattr(self.gui, 'nlp_practice_complete'):
                self.gui.nlp_practice_complete()


if __name__ == '__main__':
    try:
        node = GUINode()
        rospy.on_shutdown(node.shutdown)
        node.run()
    except rospy.ROSInterruptException:
        pass