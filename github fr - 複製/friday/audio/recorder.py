import sounddevice as sd
import numpy as np
import wave
import queue
import threading
import time

class AudioRecorder:
    """Records audio when voice is detected"""
    def __init__(self, config):
        """
        Initialize the audio recorder
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.sample_rate = config['audio'].get('sample_rate', 16000)
        self.silence_threshold = config['audio'].get('silence_threshold', 0.03)
        self.silence_duration = config['audio'].get('silence_duration', 0.5)
        self.voice_mode = False
        self.exit_flag = False
        self.audio_queue = queue.Queue()
        
    def set_voice_mode(self, enabled):
        """
        Enable or disable voice mode
        
        Args:
            enabled (bool): Whether voice mode should be enabled
        """
        self.voice_mode = enabled
        
    def start_recording(self):
        """
        Start recording audio when voice is detected
        """
        print("Audio recording thread started")
        frames = []
        is_recording = False
        silence_counter = 0
        
        def callback(indata, frames_count, time_info, status):
            nonlocal frames, is_recording, silence_counter
            
            if not self.voice_mode:
                return
                
            volume_norm = np.linalg.norm(indata) / np.sqrt(frames_count)
            
            # Detect if speaking and handle silence
            if volume_norm > self.silence_threshold:
                if not is_recording:
                    print("Voice detected, recording...")
                    is_recording = True
                frames.append(indata.copy())
                silence_counter = 0
            elif is_recording:
                frames.append(indata.copy())
                silence_counter += 1
                
                # If silence for the specified duration, stop recording
                silence_frames = int(self.silence_duration * self.sample_rate / frames_count)
                if silence_counter >= silence_frames:
                    if len(frames) > silence_frames:
                        print("Silence detected, processing speech...")
                        # Convert frames to a continuous array
                        audio_data = np.concatenate(frames, axis=0)
                        self.audio_queue.put(audio_data)
                    
                    # Reset for next recording
                    frames = []
                    is_recording = False
                    silence_counter = 0
        
        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate, 
                           blocksize=int(self.sample_rate * 0.1)):
            while not self.exit_flag:
                time.sleep(0.1)
                
    def get_audio_queue(self):
        """Get the audio queue for processing"""
        return self.audio_queue
