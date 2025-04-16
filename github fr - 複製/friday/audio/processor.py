import wave
import os
import asyncio
import numpy as np
import time

class AudioProcessor:
    """Processes audio recordings from the recorder queue"""
    def __init__(self, config, tts, rag_system):
        """
        Initialize audio processor
        
        Args:
            config (dict): Configuration dictionary
            tts (TTS): Text-to-speech system
            rag_system (RAGSystem): RAG system for generating responses
        """
        self.config = config
        self.tts = tts
        self.rag_system = rag_system
        self.sample_rate = config['audio'].get('sample_rate', 16000)
        self.exit_flag = False
        self.transcriber = None
        self.recorder = None  # Will be set later
        
    def initialize_transcriber(self):
        """Initialize the speech recognition model"""
        print("Initializing speech recognition...")
        try:
            import whisper
            model = whisper.load_model("base")
            
            def transcribe(audio_file):
                try:
                    result = model.transcribe(audio_file)
                    return {"text": result["text"]}
                except Exception as e:
                    print(f"Error transcribing audio: {e}")
                    return {"text": ""}
            
            self.transcriber = transcribe
            print("Speech recognition initialized!")
            
        except ImportError:
            print("Warning: Could not import whisper. Using external transcription.")
            
            # Define a fallback transcriber using external command
            def external_transcribe(audio_file):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["whisper", audio_file, "--model", "base", "--output_format", "txt"], 
                        capture_output=True, 
                        text=True
                    )
                    with open(f"{audio_file}.txt", "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    return {"text": text}
                except Exception as e:
                    print(f"Error with external transcription: {e}")
                    return {"text": ""}
            
            self.transcriber = external_transcribe
    
    def set_recorder(self, recorder):
        """Set the audio recorder instance"""
        self.recorder = recorder
    
    def start_processing(self):
        """Process audio from the queue and transcribe it"""
        print("Audio processing thread started")
        
        # Initialize transcriber if not already
        if self.transcriber is None:
            self.initialize_transcriber()
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.exit_flag:
            try:
                # Check if we have a recorder
                if self.recorder and not self.recorder.audio_queue.empty():
                    audio_data = self.recorder.audio_queue.get()
                    self.process_audio(audio_data, loop)
                else:
                    # Sleep a bit
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing audio: {e}")
                time.sleep(0.5)
    
    def process_audio(self, audio_data, loop):
        """
        Process a single audio clip
        
        Args:
            audio_data (np.array): Audio data to process
            loop (asyncio.EventLoop): Event loop to run async code
        """
        # Save audio to temporary file for transcription
        temp_audio_file = "temp_recording.wav"
        with wave.open(temp_audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        # Transcribe with the transcription function
        print("Transcribing speech...")
        if self.transcriber:
            result = self.transcriber(temp_audio_file)
            transcribed_text = result['text'].strip()
            
            if transcribed_text:
                print(f"\nYou said: {transcribed_text}")
                print('-' * 50)
                
                # Get response from RAG system
                response = loop.run_until_complete(
                    self.rag_system.get_answer(transcribed_text)
                )
                
                print(f"AI: {response}")
                self.tts.generate_speech(response)
                print('-' * 50)
            else:
                print("No speech detected or transcription failed.")
            
        # Clean up temporary files
        try:
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            if os.path.exists(f"{temp_audio_file}.txt"):
                os.remove(f"{temp_audio_file}.txt")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
