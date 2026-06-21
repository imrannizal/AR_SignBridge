#!/usr/bin/env python3
"""
LLM Node - Uses Gemini 2.5 Flash to convert text to sign sequences.
"""

import rospy
from std_msgs.msg import String
import requests
import os
from dotenv import load_dotenv

class LLMNode:
    def __init__(self):
        rospy.init_node('llm_node')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '..', '..', '.env')
        load_dotenv(env_path)
        
        self.API_KEY = os.getenv("GEMINI_API_KEY")
        self.API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        # Available models
        self.available_models = {
            "module1": [
                "saya_sakit_perut", "saya_demam", "saya_pening_kepala",
                "pergi_hospital", "tolong_saya", "kecemasan", "panggil_doktor"
            ],
            "module2": [
                "assalamualaikum", "apa_khabar", "waalaikumussalam", "maaf",
                "terima_kasih", "sama-sama"
            ],
            "module3": [
                "berapa_harga", "mana_tandas", "mari_makan", "mari_solat",
                "saya_mahu_balik", "saya_tidak_faham", "sekarang_waktu"
            ]
        }
        
        # Subscriber - text from GUI
        self.text_sub = rospy.Subscriber('/handsign/llm_text_input', String, self.text_callback)
        
        # Publisher - sequence to GUI
        self.sequence_pub = rospy.Publisher('/handsign/llm_sequence', String, queue_size=2)
        
        rospy.loginfo("LLM Node ready! (Gemini 2.5 Flash)")
    
    def text_callback(self, msg):
        text = msg.data.strip()
        rospy.loginfo(f"LLM input: {text}")
        
        sequence = self.call_gemini(text)
        
        if sequence:
            sequence_str = "|".join(sequence)
            rospy.loginfo(f"LLM output: {sequence_str}")
            self.sequence_pub.publish(String(data=sequence_str))
        else:
            rospy.logwarn("No valid sequence generated")
            self.sequence_pub.publish(String(data="ERROR: No matching signs"))
    
    def call_gemini(self, text):
        """Call Gemini API to convert text to sign sequence"""
        
        prompt = f"""Convert this request into sign language models to practice.

AVAILABLE MODELS (use ONLY these):
Module 1 (Medical): {', '.join(self.available_models['module1'])}
Module 2 (Greetings): {', '.join(self.available_models['module2'])}
Module 3 (General): {', '.join(self.available_models['module3'])}

RULES:
- Only use models from the available lists
- Format: module/model separated by |
- Include ALL relevant signs in order

EXAMPLES:
"I want to go home and pray" -> module3/saya_mahu_balik|module3/mari_solat
"fever and headache" -> module1/saya_demam|module1/saya_pening_kepala
"help me now" -> module1/tolong_saya|module3/sekarang_waktu
"i dont understand" -> module3/saya_tidak_faham
"stomach pain" -> module1/saya_sakit_perut
"Assalamualaikum. saya mahu makan. Terima kasih" -> module2/assalamualaikum|module3/saya_mahu_balik|module3/mari_makan|module2/terima_kasih

REQUEST: "{text}"
Return ONLY the pipe-separated sequence, no other text:"""
        
        try:
            url = f"{self.API_URL}?key={self.API_KEY}"
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(url, json=data, timeout=10)
            result = response.json()
            
            if 'candidates' in result:
                raw = result['candidates'][0]['content']['parts'][0]['text'].strip()
                rospy.loginfo(f"Gemini raw: {raw}")
                
                # Parse result
                models = [m.strip() for m in raw.split('|')]
                
                # Validate each model
                valid = []
                for m in models:
                    if '/' in m:
                        module, model = m.split('/', 1)
                        if module in self.available_models and model in self.available_models[module]:
                            valid.append(m)
                    else:
                        # Find which module has this model
                        for mod, mod_models in self.available_models.items():
                            if m in mod_models:
                                valid.append(f"{mod}/{m}")
                                break
                
                return valid
            else:
                rospy.logerr(f"Gemini error: {result}")
                return []
                
        except Exception as e:
            rospy.logerr(f"LLM error: {e}")
            return []


if __name__ == '__main__':
    try:
        LLMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass