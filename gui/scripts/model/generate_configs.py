#!/usr/bin/env python3
"""
Generate config.json files for all hand sign models.
Run once: python generate_configs.py
"""
import os
import json
from pathlib import Path

# Your sentence groups
SENTENCE_GROUPS = {
    # Module 1 - Medical Emergency
    "saya_sakit_perut": ["saya", "sakit perut"],
    "saya_demam": ["saya", "demam"],
    "saya_pening_kepala": ["saya", "sakit kepala"],
    "pergi_hospital": ["pergi", "hospital"],
    "tolong_saya": ["tolong", "saya"],
    "kecemasan": ["saya", "kecemasan"],
    "panggil_doktor": ["panggil", "doktor"],

    # Module 2 - Greeting Phrases
    "assalamualaikum": ["assalamualaikum"],
    "apa_khabar": ["apa khabar"],
    "waalaikumussalam": ["waalaikumussalam"],
    "maaf": ["maaf"],
    "terima_kasih": ["terima kasih"],
    "sama-sama": ["sama-sama"],
    
    # Module 3 - General Phrases
    "berapa_harga": ["berapa", "harga"],
    "mana_tandas": ["mana", "tandas"],
    "mari_makan": ["mari", "makan"],
    "mari_solat": ["mari", "solat"],
    "saya_mahu_balik": ["saya", "mahu", "balik"],
    "saya_tidak_faham": ["saya", "tidak", "faham"],
    "sekarang_waktu": ["sekarang", "waktu"],
}

# Which models go in which module
MODULES = {
    "module1": [
        "saya_sakit_perut", "saya_demam", "saya_pening_kepala",
        "pergi_hospital", "tolong_saya", "kecemasan", "panggil_doktor"
    ],
    "module2": [
        "assalamualaikum", "apa_khabar", "waalaikumussalam", "maaf", "sama-sama", "terima_kasih"
    ],
    "module3": [
        "berapa_harga", "mana_tandas", "mari_makan", "mari_solat",
        "saya_mahu_balik", "saya_tidak_faham", "sekarang_waktu"
    ],
}

# Base model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


def generate():
    """Generate config.json for each model."""
    
    for module_name, models in MODULES.items():
        module_path = Path(MODEL_DIR) / module_name
        module_path.mkdir(parents=True, exist_ok=True)
        
        for model_name in models:
            model_path = module_path / model_name
            model_path.mkdir(exist_ok=True)
            
            # Get classes
            if model_name in SENTENCE_GROUPS:
                classes = SENTENCE_GROUPS[model_name]
            else:
                classes = model_name.split('_')
            
            config = {
                "name": model_name,
                "display": model_name.replace('_', ' ').title(),
                "classes": classes,
                "description": f"Practice: {' -> '.join(classes)}"
            }
            
            config_path = model_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ {config_path}")
            print(f"  Classes: {classes}\n")
    
    print("Done! All config.json files created.")


if __name__ == '__main__':
    generate()