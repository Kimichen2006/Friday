import os
import asyncio
import requests
import subprocess
import sounddevice as sd
import soundfile as sf
import time
import re
import threading
import queue
import random
import numpy as np
import wave
import json
from datetime import datetime, timezone
from openai import AsyncOpenAI  # Using AsyncOpenAI for async support
from browser_use import Agent as BrowserAgent
import traceback
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Configuration variables
REFERENCE_WAV = ""
TTS_PATH = ""
GPT_MODEL = ""
SOVITS_MODEL = ""
SAMPLE_RATE = 16000
VOICE_SILENCE_THRESHOLD = 0.03
VOICE_SILENCE_DURATION = 0.5  # seconds
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = ""
KNOWLEDGE_DIR = "knowledge_base"  # Directory for your knowledge base documents
CHROMA_DB_DIR = "chroma_db"  # Directory to store the vector database
LONG_TERM_MEM_FILE = r"long_term_mem.json"
SHORT_TERM_MEM_FILE = r"short_term_memory.json"

# Global control variables
voice_mode = False
random_chat_mode = False
exit_flag = False
transcriber = None
conversation_context = []

# Queue for audio processing
audio_queue = queue.Queue()

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 角色设定提示模板
character_prompt = """You are a helpful assistant. wouldn't speak more then three sentences at a time. and will always generate (y) or (n) to decide wether to use the browser or not



chat history:
{chat_history}

context:
{context}

question: {question}

please respond as prompt said:
"""

class OpenRouterLLM:
    """OpenRouter LLM adapter class to replace Ollama"""
    def __init__(self, model=OPENROUTER_MODEL):
        self.model = model
        self.client = client
        
    async def ainvoke(self, prompt):
        """Invoke OpenRouter API asynchronously"""
        try:
            completion = await self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "AI-Girlfriend",
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            return OpenRouterResponse(completion.choices[0].message.content)
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return OpenRouterResponse("error")

class OpenRouterResponse:
    """Simple response wrapper to match the expected interface"""
    def __init__(self, content):
        self.content = content

# Replace Ollama with OpenRouter
llm = OpenRouterLLM()

async def load_short_term_memory(filename=SHORT_TERM_MEM_FILE):
    """
    Load short-term memory from file
    
    Args:
        filename (str): Path to the short-term memory file
    
    Returns:
        list: Loaded short-term memory
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"读取短期记忆出错: {e}")
        return []
    
async def async_save_short_term_memory(exchange, filename=SHORT_TERM_MEM_FILE):
    """
    Save a single exchange to short-term memory asynchronously
    
    Args:
        exchange (dict): The exchange to add to short-term memory
        filename (str): Path to save the short-term memory file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Load existing short-term memory
        memory = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
            except Exception:
                # If file exists but can't be read, start with empty memory
                memory = []
        
        # Format exchange with timestamp and default priority
        formatted_exchange = {}
        if "user" in exchange:
            formatted_exchange = {
                "user": exchange["user"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": 1  # Default priority for short-term memory
            }
        elif "bot" in exchange:
            formatted_exchange = {
                "bot": exchange["bot"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": 1  # Default priority for short-term memory
            }
        
        if formatted_exchange:
            # Add to memory and save
            memory.append(formatted_exchange)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
                
            print(f"saved to short term mem : {filename}")
    
    except Exception as e:
        print(f"error while saving short term mem: {e}")
        print(traceback.format_exc())

async def consolidate_memories():
    """
    Evaluate short-term memory priorities and consolidate with long-term memory
    """
    try:
        print("consolidating...")
        
        # Load both memory stores
        short_term = await load_short_term_memory()
        long_term = load_chat_history()  # Your existing function for long-term memory
        
        if not short_term:
            print("no short term memory consolidated needed")
            return
        
        # Reconsider priorities for all short-term memories
        for item in short_term:
            if "bot" in item:
                # Use your existing importance determination function
                priority = await determine_importance(item["bot"])
                item["priority"] = priority
        
        # Combine memories
        combined_memory = long_term + short_term
        
        # Save combined memory to long-term file
        with open(LONG_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
            json.dump(combined_memory, f, ensure_ascii=False, indent=2)
        
        # Clear short-term memory
        with open(SHORT_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        print(f"Complete! saved {len(short_term)} memories to long term mem。")
    
    except Exception as e:
        print(f"error while consolidating: {e}")
        print(traceback.format_exc())

# 读取现有的对话历史
def load_chat_history(filename=LONG_TERM_MEM_FILE):
    """
    Load chat history from file
    
    Args:
        filename (str): Path to the chat history file
    
    Returns:
        list: Loaded chat history
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"error reading chat history: {e}")
        return []

# 保存对话历史
def sync_save_chat_history(chat_history, filename=LONG_TERM_MEM_FILE):
    """
    Save chat history with timestamps and priorities
    
    Args:
        chat_history (list): List of chat exchanges
        filename (str): Path to save the chat history file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare the chat history with timestamps and priorities
        formatted_history = []
        for exchange in chat_history:
            if "user" in exchange:
                formatted_history.append({
                    "user": exchange["user"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "priority": 2  # Default user message priority (medium)
                })
            
            if "bot" in exchange:
                # Use synchronous priority determination
                priority = determine_importance(exchange["bot"])
                formatted_history.append({
                    "bot": exchange["bot"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "priority": priority
                })
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_history, f, ensure_ascii=False, indent=2)
        
        print(f"chat history saved to {filename}")
    
    except Exception as e:
        print(f"error while saving chat history: {e}")

async def trim_memory(memory_list, max_size=20000):
    """
    Trim the memory list when it exceeds the maximum size
    
    Args:
        memory_list (list): List of memory items to trim
        max_size (int): Maximum number of memory items to keep
        
    Returns:
        list: Trimmed memory list
    """
    # Check if trimming is needed
    if len(memory_list) <= max_size:
        return memory_list
    
    print(f"out of memory ({len(memory_list)}>{max_size}). trimming...")
    
    # Buffer to avoid frequent trimming
    buffer = 2000
    num_to_delete = len(memory_list) - (max_size - buffer)
    
    # Create a copy to avoid modifying during iteration
    working_memory = memory_list.copy()
    
    # Filter out memories with importance level 4 (never delete)
    permanent_memories = []
    deletable_memories = []
    
    for memory in working_memory:
        # Extract importance level, default to 2 if not found
        importance = memory.get("priority", 2)
        if importance == 4:
            permanent_memories.append(memory)
        else:
            deletable_memories.append(memory)
    
    # If nothing to delete, return original
    if not deletable_memories:
        return memory_list
    
    # Current time for age calculation
    now = datetime.now(timezone.utc)
    
    # Compute forget scores
    for memory in deletable_memories:
        # Extract timestamp and importance
        timestamp_str = memory.get("timestamp", "")
        importance = memory.get("priority", 2)
        
        # Calculate age in days
        try:
            memory_time = datetime.fromisoformat(timestamp_str)
            age_days = (now - memory_time).total_seconds() / (24 * 3600)
        except (ValueError, TypeError):
            # Default to 30 days if timestamp is invalid
            age_days = 30
        
        # Importance weights: lower importance = higher forget score
        importance_weights = {1: 5, 2: 2, 3: 1}
        weight = importance_weights.get(importance, 2)
        
        # Compute forget score
        memory["forget_score"] = weight * age_days
        
        # Add random variation for low importance memories
        if importance == 1:
            import random
            memory["forget_score"] *= random.uniform(0.9, 1.3)
    
    # Sort by forget score (descending)
    deletable_memories.sort(key=lambda x: x.get("forget_score", 0), reverse=True)
    
    # Take only what we need to delete
    to_delete = deletable_memories[:num_to_delete]
    
    # Combine permanent memories with remaining deletable memories
    retained = permanent_memories + deletable_memories[num_to_delete:]
    
    print(f"Memory: delete {len(to_delete)} memories. Prserved{len(retained)} ")
    
    return retained

# Function to periodically check and trim memory if needed
async def manage_memory_size():
    """Periodically check memory size and trim if needed"""
    global exit_flag
    
    print("manage memory function activated")
    check_interval = 600  # Check every 10 minutes
    
    while not exit_flag:
        try:
            # Load current memory
            long_term = load_chat_history()
            short_term = await load_short_term_memory()
            
            total_memory_size = len(long_term) + len(short_term)
            
            # Only trim if we have significant memory
            if total_memory_size > 20000:
                print(f"currently: {total_memory_size}....")
                
                # Trim long-term memory
                if long_term:
                    trimmed_long_term = await trim_memory(long_term)
                    
                    # Save trimmed memory
                    with open(LONG_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
                        json.dump(trimmed_long_term, f, ensure_ascii=False, indent=2)
                    
                    print(f"long term mem optimize complete: {len(long_term)} -> {len(trimmed_long_term)}")
            
            # Wait for next check
            for _ in range(check_interval):
                if exit_flag:
                    break
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"error while managing memories: {e}")
            print(traceback.format_exc())
            await asyncio.sleep(60)  # Wait before retrying

# Importance determination function
# Enhanced importance determination function without keyword section
async def determine_importance(text):
    """
    Analyze text content to determine its importance level using a local LLM or pattern matching
    
    Args:
        text (str): The text to analyze
    
    Returns:
        int: Importance level (1=low, 2=medium, 3=high, 4=permanent)
    """
    # First check for explicit markers and remove them
    if "(永久)" in text or "(4)" in text:
        return 4  # Permanent memory, never delete
    elif "(高)" in text or "(3)" in text:
        return 3
    elif "(中)" in text or "(2)" in text:
        return 2
    elif "(低)" in text or "(1)" in text:
        return 1
    
    # Use local LLM to determine importance
    try:
        from langchain_ollama import ChatOllama
        local_llm = ChatOllama(model="")  # Use your preferred local model
        
        # Construct prompt for importance analysis
        prompt = f"""
        分析下面这段文本，判断它的重要性级别:
        
        "{text}"
        
        重要性级别标准:
        - 高重要性(3): 包含情感表达、承诺、紧急事项、重要信息或与感情相关的关键内容
        - 中等重要性(2): 一般性交流和信息交换
        - 低重要性(1): 闲聊、玩笑、无关紧要的信息
        
        只回答数字: 1、2或3，表示重要性级别。
        """
        
        # Get response from local model
        response = await local_llm.ainvoke(prompt)
        content = response.content.strip()
        
        # Extract numeric value
        importance_level = re.search(r'[1-3]', content)
        if importance_level:
            return int(importance_level.group(0))
        else:
            # Fallback to default importance
            return 2
            
    except Exception as e:
        print(f"error while analyzing importance with llm: {e}")
        # Fallback based on length and complexity
        if len(text) > 100:  # Longer responses might be more important
            return 2
        return 1

# Simplified RAG System using OpenRouter embeddings
class RAGSystem:
    def __init__(self, llm, knowledge_dir, db_dir):
        self.llm = llm
        self.knowledge_dir = knowledge_dir
        self.db_dir = db_dir
        self.long_term_memory = []
        self.short_term_memory = []
        
    async def initialize(self):
        """Initialize the RAG system"""
        print("initalizing RAG system...")
        
        # Create knowledge directory if it doesn't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            print(f"created library: {self.knowledge_dir}")
            # Create example document
            with open(os.path.join(self.knowledge_dir, "sample.txt"), "w", encoding="utf-8") as f:
                f.write("this is an example.")
        
        # Load memory from both stores
        self.long_term_memory = load_chat_history()
        self.short_term_memory = await load_short_term_memory()
        print("Initialize the RAG system complete!")
    
    async def get_answer(self, query, history=None):
        """Get answer using RAG-like approach with OpenRouter"""
        try:
            # Prepare chat history context from both memory stores
            history_text = ""
            
            # Combine memories for context
            combined_memory = self.long_term_memory + self.short_term_memory
            if combined_memory:
                # Sort by timestamp if available
                try:
                    combined_memory.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                except Exception:
                    # If sorting fails, just use as is
                    pass
                
                # Get recent exchanges for context (up to 10)
                recent_history = combined_memory[-10:]
                for item in recent_history:
                    if "user" in item:
                        history_text += f"user: {item['user']}\n"
                    if "bot" in item:
                        history_text += f"friday: {item['bot']}\n"
            
            # Get knowledge context (simplified)
            context_text = ""
            try:
                # Load 1-2 random documents as context
                if os.path.exists(self.knowledge_dir):
                    files = [f for f in os.listdir(self.knowledge_dir) if f.endswith('.txt')]
                    if files:
                        sample_files = random.sample(files, min(2, len(files)))
                        for file in sample_files:
                            with open(os.path.join(self.knowledge_dir, file), 'r', encoding='utf-8') as f:
                                context_text += f.read() + "\n\n"
            except Exception as e:
                print(f"error loading dictionary: {e}")
            
            # Create prompt with context and history
            full_prompt = character_prompt.format(
                chat_history=history_text,
                context=context_text,
                question=query
            )
            
            # Get OpenRouter response
            completion = await client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "AI-Girlfriend",
                },
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            
            response = completion.choices[0].message.content
            
            # Add to short-term memory
            user_exchange = {"user": query}
            bot_exchange = {"bot": response}
            
            # Asynchronously save to short-term memory
            asyncio.create_task(async_save_short_term_memory(user_exchange))
            asyncio.create_task(async_save_short_term_memory(bot_exchange))
                
            return response
            
        except Exception as e:
            print(f"error while getting answer from RAG: {e}")
            import traceback
            print(traceback.format_exc())
            return f"error while dealing with your request: {str(e)}"

# Initialize RAG system
rag_system = RAGSystem(llm, KNOWLEDGE_DIR, CHROMA_DB_DIR)

def start_tts_service():
    """Start the TTS service"""
    os.chdir(TTS_PATH)
    subprocess.Popen([
        r'runtime\python.exe', 
        'api.py', 
        '-g', GPT_MODEL,
        '-s', SOVITS_MODEL
    ])
    print("starting TTS... ")
    time.sleep(5)  # Wait for service to start

def initialize_whisper():
    """Initialize the Whisper model for speech recognition"""
    global transcriber
    print("initialize whisper...")
    
    # Define a simple transcriber function
    def simple_transcribe(audio_file):
        try:
            # Import here to avoid initial import errors
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            return {"text": result["text"]}
        except ImportError:
            print("Can't find whisper. using plan B")
            try:
                # Fallback to external command if available
                result = subprocess.run(
                    ["whisper", audio_file, "--model", "base", "--output_format", "txt"], 
                    capture_output=True, 
                    text=True
                )
                with open(f"{audio_file}.txt", "r", encoding="utf-8") as f:
                    text = f.read().strip()
                return {"text": text}
            except:
                print("Plan B failed")
                return {"text": ""}
    
    transcriber = simple_transcribe
    print("initalizing complete!")

def generate_speech(text):
    """Generate speech from text using the TTS service"""
    # Clean text for speech by removing any importance markers
    clean_text = re.sub(r'\((?:高|中|低)\)', '', text).strip()
    
    params = {
        "refer_wav_path": REFERENCE_WAV,
        "prompt_text": "",
        "prompt_language": "",
        "text": clean_text,
        "text_language": ""
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
        print(f"errer requesting TTS: {e}")
    except Exception as e:
        print(f"error while generating speech: {e}")

async def should_use_browser(text):
    """
    Determine if the browser should be used based on (y) or (n) markers in the text
    Returns True if (y) is found, False otherwise
    """
    # Simple check for the presence of (y) marker
    if "(y)" in text:
        print("y tag detected. Using browser...")
        return True
    else:
        # Default to no browser if no marker or (n) marker
        print("n tag detected. ")
        return False

async def run_browser_search(query):
    """Run a browser search with the given query"""
    try:
        print("searching...")
        
        # Remove the (y) marker from the query if present
        clean_query = query.replace("(y)", "").strip()
        
        # Import necessary modules only when needed
        from langchain_ollama import ChatOllama
        from browser_use import Browser, BrowserConfig, Controller
        
        # Set up the local model
        local_llm = ChatOllama(model=" ") #your own local model
        
        # Configure and initialize browser
        controller = Controller()
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            )
        )
        
        # Create and run the browser agent
        browser_agent = BrowserAgent(
            task=clean_query,  # Use the cleaned query
            llm=local_llm,
            controller=controller,
            browser=browser,
        )
        
        # Execute the search
        result = await browser_agent.run()
        return result
    
    except Exception as e:
        print(f"error using browser: {e}")
        import traceback
        print(traceback.format_exc())
        return f"I tried to search information about '{query.replace('(y)', '')}', but error occured: {str(e)}"

async def get_ai_response(text):
    """Get response from AI model with RAG and browser search capability"""
    print("Friday is thinking...")
    
    try:
        # Check if browser search is needed based on (y) marker
        use_browser = await should_use_browser(text)
        
        browser_result = ""
        if use_browser:
            try:
                browser_result = await run_browser_search(text)
                print(f"browser result:\n{browser_result}\n{'='*50}")
            except Exception as browser_e:
                print(f"failed while using browser: {browser_e}")
        
        # Construct query - remove (y)/(n) markers before processing
        clean_text = re.sub(r'\([yn]\)', '', text).strip()
        
        if use_browser and browser_result:
            # Combine with search results
            query = f"""
            request: {clean_text}
            
            result:
            {browser_result}
            
            Please based on the information above and gave an answer as Friday
            """
        else:
            query = clean_text
        
        # Get RAG response
        response = await rag_system.get_answer(query)
        return response
    
    except Exception as e:
        print(f"error while Friday integrate response: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        if "ConnectionError" in str(e) or "Connection refused" in str(e):
            return "Sorry, can't connect to server. Please check if the port is activated."
        
        return "Sorry, I can't respond to that question"
    
def audio_recorder():
    """Record audio and add to queue when voice is detected"""
    global voice_mode, exit_flag
    
    print("Audio recorder on")
    frames = []
    is_recording = False
    silence_counter = 0
    
    def callback(indata, frames_count, time_info, status):
        nonlocal frames, is_recording, silence_counter
        
        if not voice_mode:
            return
            
        volume_norm = np.linalg.norm(indata) / np.sqrt(frames_count)
        
        # Detect if speaking and handle silence
        if volume_norm > VOICE_SILENCE_THRESHOLD:
            if not is_recording:
                print("Audio detected. recording...")
                is_recording = True
            frames.append(indata.copy())
            silence_counter = 0
        elif is_recording:
            frames.append(indata.copy())
            silence_counter += 1
            
            # If silence for the specified duration, stop recording
            silence_frames = int(VOICE_SILENCE_DURATION * SAMPLE_RATE / frames_count)
            if silence_counter >= silence_frames:
                if len(frames) > silence_frames:
                    print("slience detected. Thinking...")
                    # Convert frames to a continuous array
                    audio_data = np.concatenate(frames, axis=0)
                    audio_queue.put(audio_data)
                
                # Reset for next recording
                frames = []
                is_recording = False
                silence_counter = 0
    
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * 0.1)):
        while not exit_flag:
            time.sleep(0.1)

def process_audio_queue():
    """Process audio from the queue and transcribe it"""
    global exit_flag
    
    print("process audio queue started")
    while not exit_flag:
        try:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Save audio to temporary file for Whisper
                temp_audio_file = "temp_recording.wav"
                with wave.open(temp_audio_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                # Transcribe with the transcription function
                print("recording...")
                result = transcriber(temp_audio_file)
                transcribed_text = result['text'].strip()
                
                if transcribed_text:
                    print(f"\nspeak: {transcribed_text}")
                    print('-' * 50)
                    
                    # Create event loop for async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(get_ai_response(transcribed_text))
                    
                    print(f"Friday: {response}")
                    generate_speech(response)
                    print('-' * 50)
                else:
                    print("Undected or can't record")
                
                # Try to remove the temporary file
                try:
                    os.remove(temp_audio_file)
                    # Remove potential output files from whisper
                    if os.path.exists(f"{temp_audio_file}.txt"):
                        os.remove(f"{temp_audio_file}.txt")
                except:
                    pass
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"Error while processing audio: {e}")
            time.sleep(0.5)

def random_chat_thread():
    """Initiate random conversations periodically"""
    global random_chat_mode, exit_flag
    
    print("random chat is on")
    while not exit_flag:
        if random_chat_mode:
            # Wait for random time between 1-3 minutes
            wait_time = random.randint(60, 180)
            print(f"random chat will be activated after {wait_time} seconds...")
            
            # Check every second if mode has been turned off
            for _ in range(wait_time):
                if not random_chat_mode or exit_flag:
                    break
                time.sleep(1)
                
            if random_chat_mode and not exit_flag:
                print("\nrandom chat activated!")
                print('-' * 50)
                
                # Create event loop for async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(get_ai_response("say anything"))
                
                print(f"01: {response}")
                generate_speech(response)
                print('-' * 50)
        else:
            time.sleep(1)

async def main_async():
    global voice_mode, random_chat_mode, exit_flag, conversation_context, transcriber
    
    print("Starting TTS...")
    start_tts_service()
    print("TTS on!")
    
    # Initialize Whisper model
    initialize_whisper()
    
    # Initialize RAG system
    await rag_system.initialize()
    
    # Start background threads
    audio_thread = threading.Thread(target=audio_recorder)
    audio_thread.daemon = True
    audio_thread.start()
    
    processing_thread = threading.Thread(target=process_audio_queue)
    processing_thread.daemon = True
    processing_thread.start()
    
    random_chat_thread_obj = threading.Thread(target=random_chat_thread)
    random_chat_thread_obj.daemon = True
    random_chat_thread_obj.start()
    
    # Start memory management thread
    memory_management_thread = asyncio.create_task(manage_memory_size())
    
    # Add OpenRouter info
    print("\nInitalizing chat. please speak.")
    print("use 'rag rebuild' to rebuild")
    print('-' * 50)
    
    while True:
        user_input = input("\ncommand or text ('help' for options ): ")
        
        if user_input.lower() == 'quit':
            exit_flag = True
            print("正在关闭...")
            # Consolidate memories before exiting
            await consolidate_memories()
            # Cancel memory management task
            memory_management_thread.cancel()
            break
            
        elif user_input.lower() == 'help':
            print("\n可用命令:")
            print("  voice on  ")
            print("  voice off  ")
            print("  random chat on  ")
            print("  random chat off ")
            print("  rag rebuild ")
            print("  save  ")
            print("  memory trim  ")
            print("  quit        ")
            print("  help        ")
            print("  text ")
            
        elif user_input.lower() == 'voice on':
            voice_mode = True
            print("voicee is on!")
            
        elif user_input.lower() == 'voice off':
            voice_mode = False
            print("voice is off.")
            
        elif user_input.lower() == 'random chat on':
            random_chat_mode = True
            print("random chat on!")
            
        elif user_input.lower() == 'random chat off':
            random_chat_mode = False
            print("random chat off. ")
            
        elif user_input.lower() == 'rag rebuild':
            print("rebuilding...")
            # For OpenRouter, we just reinitialize the system
            await rag_system.initialize()
            print("Success!")
        
        elif user_input.lower() == 'save':
            print("saving...")
            await consolidate_memories()
            print("Saved!")
            
        elif user_input.lower() == 'memory trim':
            print("Trimming...")
            # Load, trim and save memories
            long_term = load_chat_history()
            trimmed = await trim_memory(long_term, 15000)  # Custom threshold for manual trim
            
            # Save trimmed memory
            with open(LONG_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
                
            print(f"Trimming complete: {len(long_term)} -> {len(trimmed)}")
            
        else:
            # Process text input
            try:
                print('-' * 50)
                response = await get_ai_response(user_input)
                print(f"Friday: {response}")
                generate_speech(response)
                print('-' * 50)
            except Exception as e:
                print(f"Error while thinking: {e}")

def main():
    # 确保导入json模块
    import json
    
    try:
        # Run the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\ninterrupted. shut down...")
        global exit_flag
        exit_flag = True
        # 整合记忆并保存
        try:
            # Create and run a new event loop for consolidation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(consolidate_memories())
            print("conversation saved!")
            loop.close()
        except Exception as e:
            print(f"error while saving conversation: {e}")
    except Exception as e:
        print(f"error: {e}")
        exit_flag = True
    finally:
        # Give threads time to clean up
        time.sleep(1)
    
if __name__ == '__main__':
    main()