#!/usr/bin/env python
"""
Friday - A voice-interactive AI companion using LLM and TTS
"""
# Global control variables
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
voice_mode = False
random_chat_mode = False
exit_flag = False

async def random_chat_thread(rag_system, tts):
    """Initiate random conversations periodically"""
    global random_chat_mode, exit_flag
    
    print("Random chat thread started")
    while not exit_flag:
        if random_chat_mode:
            # Wait for random time between 1-3 minutes
            import random
            wait_time = random.randint(60, 180)
            print(f"Random chat will start in {wait_time} seconds...")
            
            # Check every second if mode has been turned off
            for _ in range(wait_time):
                if not random_chat_mode or exit_flag:
                    break
                await asyncio.sleep(1)
                
            if random_chat_mode and not exit_flag:
                print("\nRandom chat initiated!")
                print('-' * 50)
                
                response = await rag_system.get_answer("随便说点什么吧")
                
                print(f"AI: {response}")
                tts.generate_speech(response)
                print('-' * 50)
        else:
            await asyncio.sleep(1)

async def main_async():
    global voice_mode, random_chat_mode, exit_flag
    
    # Load configuration
    config = load_config()
    
    print("Starting TTS service...")
    tts = TTS(config)
    tts.start_tts_service()
    print("TTS service started!")
    
    # Initialize LLM
    llm = initialize_llm(config)
    
    # Initialize RAG system
    rag_system = RAGSystem(llm, config)
    await rag_system.initialize()
    
    # Initialize audio components
    audio_recorder = AudioRecorder(config)
    audio_processor = AudioProcessor(config, tts, rag_system)
    
    # Connect recorder to processor
    audio_processor.set_recorder(audio_recorder)
    
    # Start background threads
    audio_thread = threading.Thread(target=audio_recorder.start_recording)
    audio_thread.daemon = True
    audio_thread.start()
    
    processing_thread = threading.Thread(target=audio_processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Start random chat thread
    random_chat_task = asyncio.create_task(random_chat_thread(rag_system, tts))
    
    # Start memory management thread
    memory_management_task = asyncio.create_task(manage_memory_size(config))
    
    print(f"\nChat initialized using {config['llm']['model']}. Ready for input.")
    print("Type 'rag rebuild' to rebuild knowledge base index.")
    print('-' * 50)
    
    while not exit_flag:
        user_input = input("\nCommand or text (type 'help' for options): ")
        
        if user_input.lower() == 'quit':
            exit_flag = True
            print("Shutting down...")
            # Consolidate memories before exiting
            await consolidate_memories(config)
            # Cancel tasks
            random_chat_task.cancel()
            memory_management_task.cancel()
            break
            
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  voice on   - Enable voice recognition mode")
            print("  voice off  - Disable voice recognition mode")
            print("  random chat on  - Enable random conversation initiation")
            print("  random chat off - Disable random conversation initiation")
            print("  rag rebuild - Rebuild RAG knowledge base index")
            print("  save - Manually save current conversation history")
            print("  memory trim - Manually optimize memory capacity")
            print("  quit       - Exit program")
            print("  help       - Show this help message")
            print("  (any other text) - Send as text message to AI")
            
        elif user_input.lower() == 'voice on':
            voice_mode = True
            audio_recorder.set_voice_mode(True)
            print("Voice recognition mode activated! Start speaking after this message.")
            
        elif user_input.lower() == 'voice off':
            voice_mode = False
            audio_recorder.set_voice_mode(False)
            print("Voice recognition mode deactivated.")
            
        elif user_input.lower() == 'random chat on':
            random_chat_mode = True
            print("Random chat mode activated! AI will initiate conversations randomly.")
            
        elif user_input.lower() == 'random chat off':
            random_chat_mode = False
            print("Random chat mode deactivated.")
            
        elif user_input.lower() == 'rag rebuild':
            print("Rebuilding RAG knowledge base index...")
            await rag_system.initialize()
            print("RAG index rebuilt successfully!")
        
        elif user_input.lower() == 'save':
            print("Manually consolidating and saving conversation history...")
            await consolidate_memories(config)
            print("Conversation history consolidated and saved!")
            
        elif user_input.lower() == 'memory trim':
            print("Manually optimizing memory capacity...")
            from friday.models.memory import load_chat_history, trim_memory
            # Load, trim and save memories
            long_term = load_chat_history(config)
            trimmed = await trim_memory(long_term, 15000)  # Custom threshold for manual trim
            
            # Save trimmed memory
            import json
            with open(config['memory']['long_term_file'], 'w', encoding='utf-8') as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
                
            print(f"Memory optimization complete: {len(long_term)} -> {len(trimmed)}")
            
        else:
            # Process text input
            try:
                print('-' * 50)
                response = await rag_system.get_answer(user_input)
                print(f"AI: {response}")
                tts.generate_speech(response)
                print('-' * 50)
            except Exception as e:
                print(f"Error processing text input: {e}")

def main():
    
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Girlfriend - Voice Interactive AI Companion')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.yaml')
    args = parser.parse_args()
    
    # Set config path in environment variable
    os.environ['AI_GIRLFRIEND_CONFIG'] = args.config
    
    try:
        # Run the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down...")
        global exit_flag
        exit_flag = True
        # Save memories
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(consolidate_memories(load_config()))
            print("Conversation history consolidated and saved!")
            loop.close()
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        print(traceback.format_exc())
        exit_flag = True
    finally:
        # Give threads time to clean up
        time.sleep(1)

import os
import asyncio
import threading
import time
import argparse

from friday.utils.config import load_config
from friday.audio.recorder import AudioRecorder
from friday.audio.processor import AudioProcessor
from friday.models.llm import initialize_llm
from friday.audio.tts import TTS
from friday.rag.system import RAGSystem
from friday.models.memory import consolidate_memories, manage_memory_size


if __name__ == '__main__':
    main()