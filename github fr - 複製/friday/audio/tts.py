import os
import subprocess
import time
import requests
import sounddevice as sd
import soundfile as sf
import re

class TTS:
    """Text-to-Speech service using GPT-SoVITS"""
    def __init__(self, config):
        """
        Initialize TTS service with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.service_path = config['tts'].get('service_path')
        self.reference_wav = config['tts'].get('reference_wav')
        self.gpt_model = config['tts'].get('gpt_model')
        self.sovits_model = config['tts'].get('sovits_model')
        self.prompt_text = "Hello, I am Friday, your AI assistant."
        self.prompt_language = "en"
        self.is_running = False
        
    def start_tts_service(self):
        """Start the TTS service subprocess"""
        if not self.service_path:
            print("Warning: TTS service path not configured. TTS will be non-functional.")
            return
            
        try:
            # Change to the TTS service directory
            os.chdir(self.service_path)
            
            # Start the service in a subprocess
            command = [
                'runtime\\python.exe', 
                'api.py', 
                '-g', self.gpt_model,
                '-s', self.sovits_model
            ]
            
            subprocess.Popen(command)
            print("Starting TTS service... (waiting 5 seconds for initialization)")
            time.sleep(5)  # Wait for service to start
            self.is_running = True
            
        except Exception as e:
            print(f"Error starting TTS service: {e}")
            self.is_running = False
    
    def generate_speech(self, text):
        """
        Generate speech from text using the TTS service
        
        Args:
            text (str): Text to convert to speech
        """
        if not self.is_running:
            print("Warning: TTS service is not running. Unable to generate speech.")
            return
            
        # Clean text for speech by removing any special markers
        clean_text = re.sub(r'\((?:high|medium|low)\)', '', text).strip()
        
        params = {
            "refer_wav_path": self.reference_wav,
            "prompt_text": self.prompt_text,
            "prompt_language": self.prompt_language,
            "text": clean_text,
            "text_language": self.prompt_language
        }

        try:
            response = requests.get('http://localhost:9880/', params=params)
            response.raise_for_status()

            file_path = 'temp.wav'
            with open(file_path, 'wb') as f:
                f.write(response.content)

            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
            
        except requests.exceptions.RequestException as e:
            print(f"Error requesting TTS service: {e}")
        except Exception as e:
            print(f"Error during speech generation: {e}")
