#!/usr/bin/env python3
"""
Simple Hand Sign Recognition GUI.
Practice individual handsigns one by one.
"""
import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import tkinter as tk
import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk
import threading
from pathlib import Path
import json

MODULES = {
    "Medical Emergency": {
        "folder": "module1",
        "color": "#ff4444",
        "models": [
            "saya_sakit_perut",
            "saya_demam",
            "saya_pening_kepala",
            "pergi_hospital",
            "tolong_saya",
            "kecemasan",
            "panggil_doktor",
        ]
    },
    "Greeting & Courtesies Signs": {
        "folder": "module2",
        "color": "#ff8800",
        "models": [
            "assalamualaikum",
            "apa_khabar",
            "waalaikumussalam",
            "terima_kasih",
            "maaf",
            "sama-sama",
        ]
    },
    "General Phrases": {
        "folder": "module3",
        "color": "#4488ff",
        "models": [
            "berapa_harga",
            "mana_tandas",
            "mari_makan",
            "mari_solat",
            "saya_mahu_balik",
            "saya_tidak_faham",
            "sekarang_waktu",
        ]
    }
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


def get_model_classes(module_folder, model_name):
    """Get the individual handsign classes for a model from config.json."""
    config_path = os.path.join(MODEL_DIR, module_folder, model_name, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            classes = config.get('classes', [])
            if classes:
                return classes
        except:
            pass
    
    # Fallback: split model name by underscore
    return model_name.split('_')

class SimpleHandSignGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Sign Recognition")
        self.root.geometry("900x650")
        self.root.configure(bg='#1a1a2e')

        self.root.attributes('-zoomed', True)
        
        # Callbacks for ROS
        self.on_start = None
        self.on_stop = None
        self.on_sign_change = None
        self.on_llm_text = None 
        
        # State
        self.current_module = None
        self.current_model = None
        self.classes_to_practice = []  # Individual handsigns: ["saya", "sakit_perut"]
        self.current_class_index = 0
        self.correct_count = 0
        self.running = False
        
        # Camera
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.show_menu()
    
    def show_menu(self):
        """Main menu."""
        self.running = False
        for w in self.root.winfo_children():
            w.destroy()
        
        self.root.geometry("500x450")
        self.root.configure(bg='#1a1a2e')
        
        f = tk.Frame(self.root, bg='#1a1a2e')
        f.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(f, text="HAND SIGN", font=('Arial', 24, 'bold'), 
                fg='#00ff88', bg='#1a1a2e').pack()
        tk.Label(f, text="RECOGNITION", font=('Arial', 24, 'bold'), 
                fg='#00ff88', bg='#1a1a2e').pack(pady=(0,20))
        
        tk.Button(f, text="MODULE PRACTICE", font=('Arial', 14), bg='#00ff88', fg='#000',
                 width=20, height=2, command=self.show_modules).pack(pady=5)
        tk.Button(f, text="LLM PRACTICE", font=('Arial', 14), bg='#ff8800', fg='#000',
                 width=20, height=2, command=self.show_nlp_practice).pack(pady=5)
        tk.Button(f, text="QUIT", font=('Arial', 14), bg='#ff4444', fg='#fff',
                 width=20, height=2, command=self.quit).pack(pady=5)
    
    def show_modules(self):
        """Select module."""
        for w in self.root.winfo_children():
            w.destroy()
        
        self.root.geometry("600x550")
        
        tk.Label(self.root, text="SELECT MODULE", font=('Arial', 20, 'bold'), 
                fg='#00ff88', bg='#1a1a2e').pack(pady=30)
        
        for mod_name, mod_info in MODULES.items():
            f = tk.Frame(self.root, bg='#16213e', padx=20, pady=15)
            f.pack(pady=8, fill='x', padx=50)
            
            tk.Label(f, text=mod_name, font=('Arial', 14, 'bold'), 
                    fg='#fff', bg='#16213e').pack(side='left')
            
            tk.Label(f, text=f"{len(mod_info['models'])} models", 
                    font=('Arial', 10), fg=mod_info['color'], bg='#16213e').pack(side='left', padx=10)
            
            tk.Button(f, text="SELECT", font=('Arial', 11),
                     bg='#00ff88', fg='#000', padx=15,
                     command=lambda m=mod_name: self.show_models(m)).pack(side='right')
        
        tk.Button(self.root, text="BACK TO MENU", font=('Arial', 11),
                 bg='#333', fg='#fff', padx=20, pady=8,
                 command=self.show_menu).pack(pady=20)
    
    def show_models(self, module_name):
        """Select model within module."""
        for w in self.root.winfo_children():
            w.destroy()
        
        self.current_module = module_name
        mod_info = MODULES[module_name]
        
        self.root.geometry("650x600")
        
        # Title
        tk.Label(self.root, text=module_name, font=('Arial', 20, 'bold'), 
                fg=mod_info['color'], bg='#1a1a2e').pack(pady=20)
        
        tk.Label(self.root, text="Select a model to practice:", 
                font=('Arial', 11), fg='#888', bg='#1a1a2e').pack()
        
        # Container for canvas + scrollbar
        container = tk.Frame(self.root, bg='#1a1a2e')
        container.pack(fill='both', expand=True, padx=50, pady=10)
        
        # Canvas and scrollbar
        canvas = tk.Canvas(container, bg='#1a1a2e', highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1a1a2e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create the window in canvas and center it
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Function to center the canvas window when canvas resizes
        def center_canvas_window(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
            canvas.coords(canvas_window, canvas_width // 2, 0)
        
        canvas.bind('<Configure>', center_canvas_window)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Show models - center the content
        # Create a centering frame inside scrollable_frame
        center_frame = tk.Frame(scrollable_frame, bg='#1a1a2e')
        center_frame.pack(expand=True, fill='both', padx=20)
        
        for model in mod_info['models']:
            f = tk.Frame(center_frame, bg='#16213e', padx=40, pady=10, height=130)
            f.pack(pady=4, fill='x')
            
            display = model.replace('_', ' ').title()
            classes = get_model_classes(mod_info['folder'], model)
            n_classes = len(classes)
            class_preview = ', '.join(classes[:3])
            if len(classes) > 3:
                class_preview += f" +{len(classes)-3} more"
            
            tk.Label(f, text=display, font=('Arial', 12, 'bold'), 
                    fg='#fff', bg='#16213e').pack(anchor='w')
            tk.Label(f, text=f"Signs: {class_preview}", 
                    font=('Arial', 9), fg='#ff8800', bg='#16213e').pack(anchor='w')
            tk.Label(f, text=f"{n_classes} handsigns", 
                    font=('Arial', 9), fg='#888', bg='#16213e').pack(anchor='w')
            tk.Button(f, text="PRACTICE", font=('Arial', 10),
                    bg='#00ff88', fg='#000', padx=10,
                    command=lambda m=model: self.start_practice(m)).pack(anchor='e', pady=5)
        
        # Navigation at bottom
        nav = tk.Frame(self.root, bg='#1a1a2e')
        nav.pack(side='bottom', pady=15)
        
        tk.Button(nav, text="BACK", font=('Arial', 10),
                bg='#333', fg='#fff', padx=15, pady=8,
                command=self.show_modules).pack(side='left', padx=5)
        
        tk.Button(nav, text="MENU", font=('Arial', 10),
                bg='#f44', fg='#fff', padx=15, pady=8,
                command=self.show_menu).pack(side='left', padx=5)
    
    def start_practice(self, model_name):
        """Start practicing individual handsigns one by one."""
        for w in self.root.winfo_children():
            w.destroy()
        
        self.current_model = model_name
        mod_info = MODULES[self.current_module]
        
        # Get individual handsign classes
        self.classes_to_practice = get_model_classes(mod_info['folder'], model_name)
        self.current_class_index = 0
        self.correct_count = 0
        
        self.running = True
        
        # Build practice UI
        self.root.geometry("900x650")
        self.root.configure(bg='#0a0a14')
        
        # Top bar
        bar = tk.Frame(self.root, bg='#1a1a2e', height=35)
        bar.pack(fill='x')
        bar.pack_propagate(False)
        
        self.progress_lbl = tk.Label(bar, text="", font=('Arial', 10, 'bold'), 
                                      fg='#0f8', bg='#1a1a2e')
        self.progress_lbl.pack(side='left', padx=15)
        
        tk.Label(bar, text=f"Model: {model_name.replace('_', ' ').title()}", 
                font=('Arial', 9), fg='#888', bg='#1a1a2e').pack(side='left', padx=10)
        
        self.score_lbl = tk.Label(bar, text="Correct: 0", 
                                   font=('Arial', 10), fg='#0f8', bg='#1a1a2e')
        self.score_lbl.pack(side='right', padx=15)
        
        # Camera and reference image side by side
        camera_row = tk.Frame(self.root, bg='#0a0a14')
        camera_row.pack(pady=10)

        # Camera
        cam_frame = tk.Frame(camera_row, bg='#000', width=853, height=640)
        cam_frame.pack(side='left', padx=5)
        cam_frame.pack_propagate(False)

        self.cam_lbl = tk.Label(cam_frame, bg='#111', text="CAMERA LOADING...",
                                font=('Arial', 12), fg='#444')
        self.cam_lbl.pack(fill='both', expand=True)

        # Reference image
        ref_frame = tk.Frame(camera_row, bg='#16213e', width=640, height=640)
        ref_frame.pack(side='left', padx=5)
        ref_frame.pack_propagate(False)

        tk.Label(ref_frame, text="REFERENCE", font=('Arial', 9, 'bold'),
                fg='#888', bg='#16213e').pack(pady=5)

        self.ref_lbl = tk.Label(ref_frame, bg='#16213e', text="No image",
                                font=('Arial', 10), fg='#444')
        self.ref_lbl.pack(expand=True)
        
        self.cam_lbl = tk.Label(cam_frame, bg='#111', text="CAMERA LOADING...",
                                 font=('Arial', 12), fg='#444')
        self.cam_lbl.pack(fill='both', expand=True)
        
        # Instruction card
        card = tk.Frame(self.root, bg='#16213e', padx=20, pady=15)
        card.pack(fill='x', padx=50, pady=10)
        
        # What to do
        tk.Label(card, text="DO THIS HANDSIGN:", font=('Arial', 9, 'bold'),
                fg='#ff8800', bg='#16213e').pack()
        
        self.instruction_lbl = tk.Label(card, text="", font=('Arial', 24, 'bold'),
                                         fg='#fff', bg='#16213e')
        self.instruction_lbl.pack(pady=5)
        
        # Feedback
        self.feedback_lbl = tk.Label(card, text="Show the sign above...", 
                                      font=('Arial', 12), fg='#888', bg='#16213e')
        self.feedback_lbl.pack()
        
        # Buttons
        btns = tk.Frame(self.root, bg='#0a0a14')
        btns.pack(pady=10)
        
        tk.Button(btns, text="SKIP", font=('Arial', 10), bg='#333', fg='#fff',
                 padx=15, pady=5, command=self.next_class).pack(side='left', padx=5)
        
        tk.Button(btns, text="QUIT (Q)", font=('Arial', 10), bg='#f44', fg='#fff',
                 padx=15, pady=5, command=self.stop_practice).pack(side='left', padx=5)
        
        # Keyboard shortcuts
        self.root.bind('<q>', lambda e: self.stop_practice())
        
        # Show first handsign
        self.show_current_class()
        
        # Notify ROS
        if self.on_start:
            self.on_start(mod_info['folder'], model_name)
        
        # Start camera rendering
        self.render_camera()
    
    def show_current_class(self):
        """Show current handsign class to practice."""
        if self.current_class_index >= len(self.classes_to_practice):
            self.show_complete()
            return
        
        sign = self.classes_to_practice[self.current_class_index]
        total = len(self.classes_to_practice)
        current = self.current_class_index + 1
        
        self.progress_lbl.config(text=f"Sign {current}/{total}")
        self.instruction_lbl.config(text=sign.upper())
        self.feedback_lbl.config(text="Waiting for you...", fg='#888')
        self.score_lbl.config(text=f"Correct: {self.correct_count}/{current-1}")
        
        # Notify ROS which sign we want
        if self.on_sign_change:
            self.on_sign_change(sign)
        
        # Load reference image
        ref_img = self.get_reference_image(
            MODULES[self.current_module]['folder'],
            self.current_model,
            sign
        )
        if ref_img:
            self.ref_lbl.configure(image=ref_img, text='')
            self.ref_lbl.image = ref_img
        else:
            self.ref_lbl.configure(image='', text=f"No image\nfor '{sign}'")
    
    def next_class(self):
        """Move to next handsign class."""
        self.current_class_index += 1
        self.show_current_class()
    
    def show_complete(self):
        """All handsigns practiced."""
        self.running = False
        
        if self.on_stop:
            self.on_stop()
        
        for w in self.root.winfo_children():
            w.destroy()
        
        self.root.geometry("450x400")
        self.root.configure(bg='#1a1a2e')
        
        f = tk.Frame(self.root, bg='#1a1a2e')
        f.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(f, text="MODEL COMPLETE!", font=('Arial', 22, 'bold'), 
                fg='#0f8', bg='#1a1a2e').pack(pady=10)
        
        tk.Label(f, text=self.current_model.replace('_', ' ').title(), 
                font=('Arial', 14), fg='#fff', bg='#1a1a2e').pack()
        
        total = len(self.classes_to_practice)
        tk.Label(f, text=f"Handsigns practiced: {total}", 
                font=('Arial', 12), fg='#888', bg='#1a1a2e').pack()
        
        tk.Label(f, text=f"Correct: {self.correct_count}/{total}", 
                font=('Arial', 12), fg='#0f8', bg='#1a1a2e').pack(pady=5)
        
        tk.Button(f, text="PRACTICE AGAIN", font=('Arial', 12),
                 bg='#0f8', fg='#000', padx=20, pady=10,
                 command=lambda: self.start_practice(self.current_model)).pack(pady=5)
        
        tk.Button(f, text="BACK TO MODELS", font=('Arial', 11),
                 bg='#48f', fg='#fff', padx=15, pady=8,
                 command=lambda: self.show_models(self.current_module)).pack(pady=5)
        
        tk.Button(f, text="MAIN MENU", font=('Arial', 11),
                 bg='#333', fg='#fff', padx=15, pady=8,
                 command=self.show_menu).pack(pady=5)
    
    def stop_practice(self):
        """Stop practice session."""
        self.running = False
        
        if self.on_stop:
            self.on_stop()
        
        self.root.unbind('<q>')
        self.root.unbind('<Q>')
        self.root.unbind('<space>')
        
        self.root.configure(bg='#1a1a2e')
        self.show_models(self.current_module)
    
    def update_frame(self, frame):
        """Receive camera frame from ROS."""
        with self.lock:
            if frame is not None:
                self.latest_frame = frame.copy()
    
    def update_result(self, detected_sign, confidence, is_correct):
        """Receive detection result from ROS."""
        if not self.running:
            return
        
        if is_correct:
            self.correct_count += 1
            self.feedback_lbl.config(text=f"CORRECT! ({confidence:.0%})", fg='#0f8')
            self.score_lbl.config(text=f"Correct: {self.correct_count}")
            self.root.after(1500, self.next_class)
        else:
            if confidence > 0.5:
                self.feedback_lbl.config(text=f"Try again (got: {detected_sign})", fg='#f44')
    
    def render_camera(self):
        """Render camera frame."""
        if not self.running:
            return
        
        try:
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    
                    # Get original size
                    fh, fw = frame.shape[:2]
                    
                    # Enlarge/Zoom - crop center and scale up
                    # Crop 20% from edges to zoom in
                    crop_h = int(fh * 0.8)
                    crop_w = int(fw * 0.8)
                    start_h = (fh - crop_h) // 2
                    start_w = (fw - crop_w) // 2
                    frame = frame[start_h:start_h+crop_h, start_w:start_w+crop_w]
                    
                    # Resize to display size (640x480)
                    frame = cv2.resize(frame, (853, 640))
                    
                    # Add text overlay
                    if self.current_class_index < len(self.classes_to_practice):
                        sign = self.classes_to_practice[self.current_class_index]
                        cv2.putText(frame, f"Do: {sign.upper()}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = PILImage.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    self.cam_lbl.configure(image=photo, text='')
                    self.cam_lbl.image = photo
        except:
            pass
        
        if self.running:
            self.root.after(50, self.render_camera)
    
    def quit(self):
        """Quit application."""
        self.running = False
        if self.on_stop:
            self.on_stop()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()
    
    def get_reference_image(self, module_folder, model_name, sign_name):
        """Load reference image for a handsign."""
        for ext in ['.webp', '.png', '.jpg', '.jpeg']:
            img_path = os.path.join(MODEL_DIR, module_folder, model_name, f"{sign_name}{ext}")
            if os.path.exists(img_path):
                try:
                    img = PILImage.open(img_path)
                    img = img.resize((640, 640), PILImage.Resampling.LANCZOS)  # Changed 200 to 300
                    return ImageTk.PhotoImage(img)
                except:
                    pass
        return None

    def show_nlp_practice(self):
        """NLP Practice - text input screen"""
        for w in self.root.winfo_children():
            w.destroy()
        
        self.root.geometry("600x450")
        self.root.configure(bg='#1a1a2e')
        
        f = tk.Frame(self.root, bg='#1a1a2e')
        f.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(f, text="PRACTICE", font=('Arial', 22, 'bold'),
                fg='#ff8800', bg='#1a1a2e').pack(pady=10)
        
        tk.Label(f, text="Type what you want to practice:",
                font=('Arial', 12), fg='#fff', bg='#1a1a2e').pack()
        
        self.nlp_input = tk.Entry(f, font=('Arial', 14), width=40,
                                bg='#16213e', fg='#fff', insertbackground='#fff')
        self.nlp_input.pack(pady=15, ipady=5)
        self.nlp_input.bind('<Return>', lambda e: self.send_nlp_request())
        
        tk.Label(f, text="Quick examples:", font=('Arial', 10, 'bold'),
                fg='#888', bg='#1a1a2e').pack(pady=(10,5))
        
        examples = [
            "I want to go home and pray",
            "Fever and headache",
            "Help me now",
            "saya tidak makan",
            "saya sakit perut"
        ]
        
        for ex in examples:
            tk.Button(f, text=ex, font=('Arial', 9),
                    bg='#333', fg='#fff', width=35,
                    command=lambda t=ex: self.nlp_input.insert(0, t)).pack(pady=2)
        
        btn_frame = tk.Frame(f, bg='#1a1a2e')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="SEND", font=('Arial', 12, 'bold'),
                bg='#00ff88', fg='#000', padx=20, pady=8,
                command=self.send_nlp_request).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="BACK", font=('Arial', 12),
                bg='#333', fg='#fff', padx=20, pady=8,
                command=self.show_menu).pack(side='left', padx=5)
        
        self.nlp_status = tk.Label(f, text="", font=('Arial', 11),
                                    fg='#0f8', bg='#1a1a2e')
        self.nlp_status.pack(pady=5)
        
        self.nlp_input.focus_set()

    def send_nlp_request(self):
        """Send text to NLP node"""
        text = self.nlp_input.get().strip()
        if not text:
            return
        self.nlp_status.config(text="Matching signs...", fg='#ff8800')
        if self.on_llm_text:
            self.on_llm_text(text)

    def show_nlp_practice_session(self, signs):
        """Show NLP practice with camera"""
        for w in self.root.winfo_children():
            w.destroy()
        
        self.running = True
        self.nlp_signs = signs
        self.nlp_current_idx = 0
        self.nlp_correct = 0
        
        self.root.geometry("900x700")
        self.root.configure(bg='#0a0a14')
        
        # Top bar
        bar = tk.Frame(self.root, bg='#1a1a2e', height=40)
        bar.pack(fill='x')
        
        tk.Label(bar, text="NLP PRACTICE", font=('Arial', 12, 'bold'),
                fg='#ff8800', bg='#1a1a2e').pack(side='left', padx=15)
        
        self.nlp_progress_lbl = tk.Label(bar, text=f"Sign 1/{len(signs)}",
                                        font=('Arial', 10), fg='#0f8', bg='#1a1a2e')
        self.nlp_progress_lbl.pack(side='right', padx=15)
        
        # Sequence
        seq_frame = tk.Frame(self.root, bg='#16213e', padx=10, pady=5)
        seq_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(seq_frame, text="Sequence:", font=('Arial', 9, 'bold'),
                fg='#888', bg='#16213e').pack(side='left', padx=5)
        
        seq_text = " → ".join(signs)
        tk.Label(seq_frame, text=seq_text, font=('Arial', 10),
                fg='#fff', bg='#16213e').pack(side='left', padx=5)
        
        # Camera
        cam_frame = tk.Frame(self.root, bg='#000', width=640, height=480)
        cam_frame.pack(pady=10)
        
        self.cam_lbl = tk.Label(cam_frame, bg='#111', text="CAMERA LOADING...")
        self.cam_lbl.pack()
        
        # Instruction
        card = tk.Frame(self.root, bg='#16213e', padx=20, pady=15)
        card.pack(fill='x', padx=50, pady=10)
        
        tk.Label(card, text="DO THIS SIGN:", font=('Arial', 9, 'bold'),
                fg='#ff8800', bg='#16213e').pack()
        
        self.nlp_instruction = tk.Label(card, text=signs[0].upper(),
                                        font=('Arial', 28, 'bold'),
                                        fg='#fff', bg='#16213e')
        self.nlp_instruction.pack(pady=5)
        
        self.nlp_feedback = tk.Label(card, text="Show the sign...",
                                    font=('Arial', 12), fg='#888', bg='#16213e')
        self.nlp_feedback.pack()
        
        # Progress dots
        dots_frame = tk.Frame(self.root, bg='#0a0a14')
        dots_frame.pack(pady=5)
        
        self.nlp_dots = []
        for i in range(len(signs)):
            dot = tk.Label(dots_frame, text="○", font=('Arial', 14),
                        fg='#444', bg='#0a0a14')
            dot.pack(side='left', padx=3)
            self.nlp_dots.append(dot)
        self.nlp_dots[0].config(text="●", fg='#ff8800')
        
        # Buttons
        btns = tk.Frame(self.root, bg='#0a0a14')
        btns.pack(pady=10)
        
        tk.Button(btns, text="QUIT (Q)", font=('Arial', 10), bg='#f44', fg='#fff',
                padx=20, pady=8, command=self.stop_nlp_practice).pack()
        
        self.root.bind('<q>', lambda e: self.stop_nlp_practice())
        
        self.render_camera()

    def update_nlp_progress(self, index, correct):
        """Update NLP progress"""
        if correct:
            self.nlp_correct += 1
            self.nlp_dots[index].config(text="✓", fg='#0f8')
            self.nlp_feedback.config(text="CORRECT! ✓", fg='#0f8')
            
            if index + 1 < len(self.nlp_signs):
                self.nlp_current_idx = index + 1
                next_sign = self.nlp_signs[index + 1]
                self.nlp_instruction.config(text=next_sign.upper())
                self.nlp_progress_lbl.config(text=f"Sign {index+2}/{len(self.nlp_signs)}")
                self.nlp_dots[index + 1].config(text="●", fg='#ff8800')
        else:
            self.nlp_feedback.config(text="Try again...", fg='#f44')

    def nlp_practice_complete(self):
        """NLP complete screen"""
        self.running = False
        
        for w in self.root.winfo_children():
            w.destroy()
        
        self.root.geometry("450x400")
        self.root.configure(bg='#1a1a2e')
        
        f = tk.Frame(self.root, bg='#1a1a2e')
        f.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(f, text="PRACTICE COMPLETE!", font=('Arial', 22, 'bold'),
                fg='#0f8', bg='#1a1a2e').pack(pady=10)
        
        seq = " → ".join(self.nlp_signs)
        tk.Label(f, text=seq, font=('Arial', 12), fg='#fff', bg='#1a1a2e').pack()
        
        tk.Label(f, text=f"Correct: {self.nlp_correct}/{len(self.nlp_signs)}",
                font=('Arial', 14), fg='#0f8', bg='#1a1a2e').pack(pady=10)
        
        tk.Button(f, text="NEW PRACTICE", font=('Arial', 12),
                bg='#0f8', fg='#000', padx=20, pady=10,
                command=self.show_nlp_practice).pack(pady=5)
        
        tk.Button(f, text="MAIN MENU", font=('Arial', 11),
                bg='#333', fg='#fff', padx=15, pady=8,
                command=self.show_menu).pack(pady=5)

    def stop_nlp_practice(self):
        """Stop NLP practice"""
        self.running = False
        if self.on_stop:
            self.on_stop()
        self.root.unbind('<q>')
        self.show_menu()

if __name__ == '__main__':
    app = SimpleHandSignGUI()
    app.run()