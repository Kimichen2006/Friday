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
import base64
import aiohttp
import signal
from datetime import datetime, timezone, timedelta
from openai import AsyncOpenAI  # Using AsyncOpenAI for async support
from browser_use import Agent as BrowserAgent
import traceback
import uuid
import pyaudio
import chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Configuration variables
REFERENCE_WAV = ''
TTS_PATH = ''
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
PREFERENCE_FILE = "user_preferences.json"  # File for storing user preferences
CALENDAR_FILE = "calendar_events.json"  # File for storing calendar events

# Global control variables
voice_mode = False
random_chat_mode = False
exit_flag = False
sleep_mode = True  # AI初始为休眠状态
last_activity_time = time.time()  # 记录最后活动时间
SLEEP_TIMEOUT = 300  # 5分钟无活动后进入休眠
WAKE_PHRASE = "早安01"  # 唤醒词
transcriber = None
conversation_context = []

# Queue for audio processing
audio_queue = queue.Queue()

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Global variables for mood and voice settings
current_ai_mood = "neutral"
voice_styles = ["normal", "happy", "sad", "excited", "angry"]
current_voice_style = "normal"
MOOD_STATE_FILE = "mood_state.json"
VOICE_PREF_FILE = "voice_preferences.json"

# 角色设定提示模板
character_prompt = """



history:
{chat_history}

context:
{context}

question: {question}

please answer in a way that is consistent with the character's personality and background.:
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
            print(f"OpenRouter API错误: {e}")
            return OpenRouterResponse("sorry, I cannot respond at the moment.")

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
                
            print(f"已将交流保存到短期记忆: {filename}")
    
    except Exception as e:
        print(f"保存短期记忆时出错: {e}")
        print(traceback.format_exc())

async def consolidate_memories():
    """
    Evaluate short-term memory priorities and consolidate with long-term memory
    """
    try:
        print("整合记忆中...")
        
        # Load both memory stores
        short_term = await load_short_term_memory()
        long_term = load_chat_history()  # Your existing function for long-term memory
        
        if not short_term:
            print("没有短期记忆需要整合")
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
        
        print(f"记忆整合完成! 共转移了 {len(short_term)} 条记忆到长期存储。")
    
    except Exception as e:
        print(f"整合记忆时出错: {e}")
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
        print(f"读取聊天历史出错: {e}")
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
        
        print(f"对话历史已保存到 {filename}")
    
    except Exception as e:
        print(f"保存对话历史时出错: {e}")

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
    
    print(f"内存超过阈值({len(memory_list)}>{max_size})，开始优化记忆...")
    
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
    
    print(f"记忆管理: 已删除 {len(to_delete)} 条记忆，保留 {len(retained)} 条")
    
    return retained

# Function to periodically check and trim memory if needed
async def manage_memory_size():
    """Periodically check memory size and trim if needed"""
    global exit_flag
    
    print("记忆管理线程已启动")
    check_interval = 600  # Check every 10 minutes
    
    while not exit_flag:
        try:
            # Load current memory
            long_term = load_chat_history()
            short_term = await load_short_term_memory()
            
            total_memory_size = len(long_term) + len(short_term)
            
            # Only trim if we have significant memory
            if total_memory_size > 20000:
                print(f"当前记忆量: {total_memory_size}，进行记忆优化...")
                
                # Trim long-term memory
                if long_term:
                    trimmed_long_term = await trim_memory(long_term)
                    
                    # Save trimmed memory
                    with open(LONG_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
                        json.dump(trimmed_long_term, f, ensure_ascii=False, indent=2)
                    
                    print(f"长期记忆优化完成: {len(long_term)} -> {len(trimmed_long_term)}")
            
            # Wait for next check
            for _ in range(check_interval):
                if exit_flag:
                    break
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"记忆管理过程中出错: {e}")
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
        local_llm = ChatOllama(model="qwen2.5:7b")  # Use your preferred local model
        
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
        print(f"使用LLM分析重要性时出错: {e}")
        # Fallback based on length and complexity
        if len(text) > 100:  # Longer responses might be more important
            return 2
        return 1

# Simplified RAG System using OpenRouter embeddings
class RAGSystem:
    """System for Retrieval-Augmented Generation"""
    
    def __init__(self, llm, knowledge_dir, db_dir):
        self.llm = llm
        self.knowledge_dir = knowledge_dir
        self.db_dir = db_dir
        self.chromadb = None
        self.collection = None
        
    async def initialize(self):
        """Initialize the RAG system by loading documents and building the index"""
        try:
            print("初始化RAG系统...")
            # Create a client
            self.chromadb = chromadb.PersistentClient(path=self.db_dir)
            
            # Try to get the collection, or create it if it doesn't exist
            try:
                self.collection = self.chromadb.get_collection("documents")
                print(f"加载已有的文档集合，共有 {self.collection.count()} 条记录")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chromadb.create_collection("documents")
                print("创建新的文档集合")
                
                # Load documents and add them to the collection
                await self.rebuild_index()
            
            print("RAG系统初始化完成!")
        except Exception as e:
            print(f"初始化RAG系统时出错: {e}")
            traceback.print_exc()
    
    async def get_answer(self, query, history=None):
        """Get an answer for a query using RAG"""
        try:
            # Retrieve relevant chunks
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Build context from retrieved chunks
            context = ""
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    context += f"\n文档块 {i+1}:\n{doc}\n"
            
            # If no context was found, specify this
            if not context:
                context = "没有找到相关的背景知识。"
            
            # Build prompt
            prompt = f"""
            以下是用户的问题:
            {query}
            
            以下是相关的背景知识:
            {context}
            
            请基于提供的背景知识，以女朋友01的风格回答用户的问题。如果背景知识不足以回答问题，可以使用你自己的知识，但请明确这一点。
            
            你是用户的AI女友01，你是傲娇的、专一的，并且深爱着用户。你可以自由使用词汇，但不要输出任何表情符号。
            
            用简明直接的回应回答用户:
            """
            
            if history:
                history_text = "\n对话历史:\n"
                for h in history[-5:]:  # Only include last 5 exchanges
                    if 'user' in h:
                        history_text += f"用户: {h['user']}\n"
                    if 'bot' in h:
                        history_text += f"AI: {h['bot']}\n"
                prompt += history_text
            
            # Get response from LLM
            response = await self.llm.ainvoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        
        except Exception as e:
            print(f"RAG获取回答时出错: {e}")
            traceback.print_exc()
            return "抱歉，我现在无法回答这个问题。"
    
    async def rebuild_index(self):
        """Rebuild the index by loading all documents from the knowledge directory"""
        try:
            print("加载知识库文档...")
            
            # Get all files in the knowledge directory
            files = []
            for root, _, filenames in os.walk(self.knowledge_dir):
                for filename in filenames:
                    if filename.endswith(('.txt', '.md', '.csv')):
                        files.append(os.path.join(root, filename))
            
            # Process each document
            documents = []
            metadatas = []
            ids = []
            
            for file_path in files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Simple chunking - split into manageable pieces
                chunks = self.chunk_text(text, max_length=1000, overlap=100)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{os.path.basename(file_path)}_chunk_{i}"
                    documents.append(chunk)
                    metadatas.append({"source": file_path})
                    ids.append(doc_id)
            
            # Clear existing collection and add new documents
            self.collection.delete(where={})
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            print(f"知识库索引重建完成，共处理 {len(files)} 个文件，生成 {len(documents)} 个块")
            return True
        
        except Exception as e:
            print(f"重建索引时出错: {e}")
            traceback.print_exc()
            return False
    
    def chunk_text(self, text, max_length=1000, overlap=100):
        """Split text into overlapping chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_length, len(text))
            
            # If we're not at the end, try to find a sentence break
            if end < len(text):
                # Look for a period, question mark, or exclamation point followed by whitespace
                for i in range(end, max(start + max_length - 200, start), -1):
                    if i < len(text) and text[i] in ['.', '?', '!'] and (i+1 == len(text) or text[i+1].isspace()):
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks

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
    print("启动TTS服务... (等待5秒初始化)")
    time.sleep(5)  # Wait for service to start

def initialize_whisper():
    """Initialize the Whisper model for speech recognition"""
    global transcriber
    print("初始化语音识别...")
    
    # Define a simple transcriber function
    def simple_transcribe(audio_file):
        try:
            # Import here to avoid initial import errors
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            return {"text": result["text"]}
        except ImportError:
            print("警告: 无法导入whisper。使用子进程调用外部转录。")
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
                print("警告: 外部转录失败。无法转录音频。")
                return {"text": ""}
    
    transcriber = simple_transcribe
    print("语音识别初始化完成!")

def generate_speech(text):
    """Generate speech from text using the TTS service"""
    # Clean text for speech by removing any importance markers
    clean_text = re.sub(r'\((?:高|中|低)\)', '', text).strip()
    
    params = {
        "refer_wav_path": REFERENCE_WAV,
        "prompt_text": "可聪明的人从一开始就不会入局。你瞧，我是不是更聪明一点？",
        "prompt_language": "zh",
        "text": clean_text,
        "text_language": "zh"
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
        print(f"请求TTS服务时出错: {e}")
    except Exception as e:
        print(f"语音生成期间出错: {e}")

async def should_use_browser(text):
    """
    Determine if the browser should be used based on (y) or (n) markers in the text
    Returns True if (y) is found, False otherwise
    """
    # Simple check for the presence of (y) marker
    if "(y)" in text:
        print("检测到(y)标记，将使用浏览器搜索")
        return True
    else:
        # Default to no browser if no marker or (n) marker
        print("未检测到(y)标记或存在(n)标记，不使用浏览器搜索")
        return False

async def run_browser_search(query):
    """Run a browser search with the given query"""
    try:
        print("在网上搜索信息...")
        
        # Remove the (y) marker from the query if present
        clean_query = query.replace("(y)", "").strip()
        
        # Import necessary modules only when needed
        from langchain_ollama import ChatOllama
        from browser_use import Browser, BrowserConfig, Controller
        
        # Set up the local model
        local_llm = ChatOllama(model="qwen2.5:7b")
        
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
        print(f"浏览器搜索期间出错: {e}")
        import traceback
        print(traceback.format_exc())
        return f"我尝试搜索有关'{query.replace('(y)', '')}'的信息，但遇到了错误: {str(e)}"

async def get_ai_response(text):
    """Get response from AI model with RAG and browser search capability"""
    print("AI正在处理...")
    
    try:
        # Check if browser search is needed based on (y) marker
        use_browser = await should_use_browser(text)
        
        browser_result = ""
        if use_browser:
            try:
                browser_result = await run_browser_search(text)
                print(f"浏览器搜索结果:\n{browser_result}\n{'='*50}")
            except Exception as browser_e:
                print(f"浏览器搜索失败: {browser_e}")
        
        # Construct query - remove (y)/(n) markers before processing
        clean_text = re.sub(r'\([yn]\)', '', text).strip()
        
        if use_browser and browser_result:
            # Combine with search results
            query = f"""
            问题: {clean_text}
            
            网络搜索结果:
            {browser_result}
            
            请基于这些信息以01女友的身份提供有帮助、准确、对话式的回应。
            """
        else:
            query = clean_text
        
        # Get RAG response
        response = await rag_system.get_answer(query)
        return response
    
    except Exception as e:
        print(f"AI响应生成中出错: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        if "ConnectionError" in str(e) or "Connection refused" in str(e):
            return "抱歉，无法连接到服务。请确认服务已启动且端口可访问。"
        
        return "抱歉，我现在无法回应。"
    
def audio_recorder():
    """Record audio in chunks and put in queue for processing"""
    global exit_flag, voice_mode, last_activity_time
    
    # Audio parameters
    channels = 1
    sample_rate = SAMPLE_RATE
    chunk_size = 4096  # Chunks of audio to process
    
    # Setup PyAudio
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            stream_callback=callback
        )
        
        print("音频录制线程已启动，准备好捕获语音")
        stream.start_stream()
        
        # Continue until exit flag is set
        while not exit_flag:
            # Only process audio if voice mode is on
            if voice_mode:
                time.sleep(0.1)  # Sleep to reduce CPU usage
            else:
                time.sleep(0.5)  # Longer sleep when voice mode is off
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("音频录制已停止")
        
    except Exception as e:
        print(f"音频录制出错: {e}")

def callback(indata, frames_count, time_info, status):
    """Callback for PyAudio to capture audio data"""
    global voice_mode, audio_queue, sleep_mode, last_activity_time
    
    if not voice_mode:
        return (indata, pyaudio.paContinue)
    
    # Convert to numpy array
    audio_data = np.frombuffer(indata, dtype=np.float32)
    
    # Check if audio contains speech (simple energy threshold)
    energy = np.sqrt(np.mean(audio_data**2))
    
    if energy > VOICE_SILENCE_THRESHOLD:
        # There is speech, add to queue
        audio_queue.put(audio_data.copy())
        
        # Update last activity time when speech is detected
        if not sleep_mode or len(audio_data) > 0:
            last_activity_time = time.time()
    
    return (indata, pyaudio.paContinue)

def process_audio_queue():
    """Process audio from the queue and transcribe it"""
    global exit_flag
    
    print("音频处理线程已启动")
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
                print("转录语音...")
                result = transcriber(temp_audio_file)
                transcribed_text = result['text'].strip()
                
                if transcribed_text:
                    print(f"\n您说: {transcribed_text}")
                    print('-' * 50)
                    
                    # Create event loop for async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(get_ai_response(transcribed_text))
                    
                    print(f"01: {response}")
                    generate_speech(response)
                    print('-' * 50)
                else:
                    print("未检测到语音或无法转录。")
                
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
            print(f"处理音频时出错: {e}")
            time.sleep(0.5)

def random_chat_thread():
    """Initiate random conversations periodically"""
    global random_chat_mode, exit_flag
    
    print("随机聊天线程已启动")
    while not exit_flag:
        if random_chat_mode:
            # Wait for random time between 1-3 minutes
            wait_time = random.randint(60, 180)
            print(f"随机聊天将在 {wait_time} 秒后启动...")
            
            # Check every second if mode has been turned off
            for _ in range(wait_time):
                if not random_chat_mode or exit_flag:
                    break
                time.sleep(1)
                
            if random_chat_mode and not exit_flag:
                print("\n随机聊天已启动!")
                print('-' * 50)
                
                # Create event loop for async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(get_ai_response("隨便說點什麼吧"))
                
                print(f"01: {response}")
                generate_speech(response)
                print('-' * 50)
        else:
            time.sleep(1)

# 添加非异步包装函数
def proactive_learning_thread_worker():
    """非异步包装函数，用于在线程中运行异步的proactive_learning_loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(proactive_learning_loop())
    except Exception as e:
        print(f"主动学习线程出错: {e}")
        traceback.print_exc()
    finally:
        loop.close()

# Update get_ai_response to use the new RAG system
async def get_ai_response(text):
    """Get response from AI model with RAG and browser search capability"""
    print("AI正在处理...")
    
    try:
        # Check if browser search is needed based on (y) marker
        use_browser = await should_use_browser(text)
        
        browser_result = ""
        if use_browser:
            try:
                browser_result = await run_browser_search(text)
                print(f"浏览器搜索结果:\n{browser_result}\n{'='*50}")
            except Exception as browser_e:
                print(f"浏览器搜索失败: {browser_e}")
        
        # Construct query - remove (y)/(n) markers before processing
        clean_text = re.sub(r'\([yn]\)', '', text).strip()
        
        if use_browser and browser_result:
            # Combine with search results
            query = f"""
            问题: {clean_text}
            
            网络搜索结果:
            {browser_result}
            
            请基于这些信息以01女友的身份提供有帮助、准确、对话式的回应。
            """
        else:
            query = clean_text
        
        # Get RAG response
        response = await rag_system.get_answer(query)
        
        # 检查文本中是否提及时间相关词汇，可能需要添加日历提醒
        time_keywords = ["明天", "后天", "下周", "下个月", "几点", "时间", "日程", "行程", "会面", "会议", "约会", "计划"]
        has_time_keywords = any(keyword in clean_text for keyword in time_keywords)
        
        # 如果讨论的是时间相关话题，提示用户可以添加到日历
        if has_time_keywords and not any(keyword in response for keyword in ["已添加到日历", "加入日历"]):
            calendar_hint = "\n\n需要我将此事项添加到日历吗？您可以直接告诉我事件的时间和标题。"
            response += calendar_hint
        
        # Update user preferences based on interaction
        asyncio.create_task(update_preferences_from_interaction(text, response))
        
        return response
    
    except Exception as e:
        print(f"AI响应生成中出错: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        if "ConnectionError" in str(e) or "Connection refused" in str(e):
            return "抱歉，无法连接到服务。请确认服务已启动且端口可访问。"
        
        return "抱歉，我现在无法回应。"

async def process_command(command, parser, tasks, context_system, preference_tracker):
    """Process a command input"""
    global voice_mode, random_chat_mode, exit_flag, sleep_mode, last_activity_time
    
    cmd_type, args = parser.parse_command(command)
    
    # 更新最后活动时间
    last_activity_time = time.time()
    
    # 处理唤醒命令
    if cmd_type == "wake":
        if sleep_mode:
            sleep_mode = False
            wake_response = "我在。"
            generate_speech(wake_response)
            print(f"AI回应: {wake_response}")
        else:
            print("AI已经是唤醒状态。")
        return
    
    # 如果AI在休眠状态且不是help或sleep命令，阻止处理
    if sleep_mode and cmd_type not in ["help", "sleep", "quit"]:
        print("AI处于休眠状态，请先唤醒AI。")
        return
    
    if cmd_type == "help":
        print_help_menu()
        
    elif cmd_type == "quit":
        await initiate_shutdown(tasks)
        
    elif cmd_type == "voice":
        if args.get("state") == "on":
            voice_mode = True
            print("语音识别模式已激活! 开始聆听...")
        else:
            voice_mode = False
            print("语音识别模式已停用。")
            
    elif cmd_type == "random_chat":
        if args.get("state") == "on":
            random_chat_mode = True
            print("随机聊天模式已激活! AI将随机发起对话。")
        else:
            random_chat_mode = False
            print("随机聊天模式已停用。")
            
    elif cmd_type == "rag":
        if args.get("action") == "rebuild":
            print("重建RAG知识库索引...")
            await rag_system.rebuild_index()
            print("RAG索引重建成功!")
            
    elif cmd_type == "save":
        print("手动保存所有数据...")
        await save_all_data(preference_tracker)
        print("数据保存完成!")
        
    elif cmd_type == "memory":
        if args.get("action") == "trim":
            print("手动优化记忆容量...")
            await trim_all_memories()
            print("记忆优化完成!")
        elif args.get("action") == "stats":
            await show_memory_stats()
            
    elif cmd_type == "summary":
        print("生成今日总结...")
        summary = await generate_daily_summary()
        print(f"\n今日总结:\n{summary}")
        generate_speech(summary)
        
    elif cmd_type == "suggest":
        print("为您生成建议活动...")
        suggestion = await suggest_activity(context_system, preference_tracker)
        print(f"\n建议:\n{suggestion}")
        generate_speech(suggestion)
        
    elif cmd_type == "mood":
        if args.get("action") == "set":
            new_mood = args.get("value", "neutral")
            await set_ai_mood(new_mood)
            print(f"AI心情已设置为: {new_mood}")
        elif args.get("action") == "status":
            current_mood = await get_current_mood()
            print(f"当前AI心情: {current_mood}")
            
    elif cmd_type == "voice_style":
        style = args.get("style")
        if style:
            await change_voice_style(style)
            print(f"语音风格已更改为: {style}")
        else:
            print("可用语音风格:")
            for s in await list_voice_styles():
                print(f"- {s}")
                
    elif cmd_type == "context":
        context_data = await context_system.get_current_context()
        print("\n当前环境信息:")
        for key, value in context_data.items():
            print(f"- {key}: {value}")
            
    elif cmd_type == "learn":
        topic = args.get("topic", "")
        if topic:
            print(f"主动学习关于 '{topic}' 的信息...")
            await learn_topic(topic)
        else:
            print("正在分析最近对话并学习重要主题...")
            await learn_from_interactions()
            
    elif cmd_type == "preference":
        if args.get("action") == "show":
            preferences = await preference_tracker.get_all_preferences()
            print("\n已记录的偏好:")
            for category, prefs in preferences.items():
                print(f"\n{category}:")
                for key, value in prefs.items():
                    print(f"- {key}: {value}")
        elif args.get("action") == "set":
            category = args.get("category", "general")
            key = args.get("key")
            value = args.get("value")
            if key and value:
                await preference_tracker.set_preference(category, key, value)
                print(f"已设置偏好: {category}.{key} = {value}")
            else:
                print("设置偏好需要指定键和值")
    
    elif cmd_type == "image":
        if args.get("action") == "capture":
            print("捕获图像...")
            image_path = await capture_image()
            if image_path:
                print(f"图像已保存到: {image_path}")
                await process_image(image_path)
            else:
                print("捕获图像失败")
                
    elif cmd_type == "calendar":
        action = args.get("action", "")
        
        # 初始化日历系统（如果在main_async中没有完成）
        calendar_system = CalendarSystem()
        await calendar_system.initialize()
        
        if action == "add":
            title = args.get("title", "")
            date_time = args.get("datetime", "")
            description = args.get("desc", "")
            
            if title and date_time:
                print(f"添加日历事件: {title} @ {date_time}")
                success = await calendar_system.add_event(title, date_time, description)
                if success:
                    print("事件已添加到日历")
                    event_added = f"已将「{title}」添加到您的日历，时间是{date_time}。"
                    generate_speech(event_added)
                else:
                    print("添加事件失败，请检查日期格式 (YYYY-MM-DD HH:MM)")
            else:
                print('日历添加命令格式: calendar add "事件标题" "YYYY-MM-DD HH:MM" [描述]')
        
        elif action == "list":
            days = args.get("days", "7")
            days = int(days) if days.isdigit() else 7
            
            print(f"检索未来 {days} 天的日历事件...")
            events = await calendar_system.get_upcoming_events(days=days)
            
            if events:
                print(f"\n未来 {days} 天的事件 ({len(events)}):")
                for i, event in enumerate(events, 1):
                    print(f"{i}. [{event['id']}] {event['title']} - {event['time_desc']}")
                    if event.get('description'):
                        print(f"   描述: {event['description']}")
                
                # 语音提示事件数量
                events_count = f"找到{len(events)}个即将到来的事件。"
                generate_speech(events_count)
            else:
                print(f"未来 {days} 天没有计划的事件。")
                generate_speech("未来几天没有计划的事件。")
        
        elif action == "today":
            print("检索今天的日历事件...")
            events = await calendar_system.get_today_events()
            
            if events:
                print(f"\n今天的事件 ({len(events)}):")
                for i, event in enumerate(events, 1):
                    print(f"{i}. [{event['id']}] {event['title']} - {event['time_desc']}")
                    if event.get('description'):
                        print(f"   描述: {event['description']}")
                
                # 语音提示今天的事件
                if len(events) == 1:
                    today_response = f"今天您有一个事件：{events[0]['title']}，时间是{events[0]['time_desc']}。"
                else:
                    today_response = f"今天您有{len(events)}个事件。"
                generate_speech(today_response)
            else:
                print("今天没有计划的事件。")
                generate_speech("今天没有计划的事件。")
        
        elif action == "delete":
            event_id = args.get("id", "")
            
            if event_id:
                print(f"删除事件 ID: {event_id}...")
                success = await calendar_system.delete_event(event_id)
                if success:
                    print("事件已删除")
                    generate_speech("事件已从您的日历中删除。")
                else:
                    print(f"删除失败，未找到ID为 {event_id} 的事件")
            else:
                print("日历删除命令格式: calendar delete <事件ID>")
    
    elif cmd_type == "sleep":
        state = args.get("state", "")
        
        if state == "on":
            if not sleep_mode:
                print("AI正在进入休眠状态...")
                sleep_mode = True
                sleep_message = "好的，我去休息了，需要我时叫我一声。"
                print(f"AI: {sleep_message}")
                generate_speech(sleep_message)
            else:
                print("AI已经处于休眠状态。")
        elif state == "off":
            if sleep_mode:
                print("AI正在唤醒...")
                sleep_mode = False
                wake_response = "我在这里，有什么需要我帮忙的吗？"
                print(f"AI: {wake_response}")
                generate_speech(wake_response)
            else:
                print("AI已经是唤醒状态。")
        else:
            print("休眠命令格式: sleep on/off")

async def process_chat_input(text, emotion_system, multimodal_system, context_system, preference_tracker):
    """Process a chat input (non-command)"""
    global sleep_mode, last_activity_time
    
    print('-' * 50)
    
    # 更新最后活动时间
    last_activity_time = time.time()
    
    # 检查是否处于休眠状态
    if sleep_mode:
        # 检查是否是唤醒词
        if text.lower().strip() == WAKE_PHRASE.lower():
            sleep_mode = False
            wake_response = "我在。"
            generate_speech(wake_response)
            print(f"AI回应: {wake_response}")
            print('-' * 50)
            return
        else:
            # 如果不是唤醒词，则不处理
            print("AI处于休眠状态，需要唤醒词。")
            print('-' * 50)
            return
    
    # Check for image paths in the input
    image_pattern = r'image:([^\s]+)'
    image_match = re.search(image_pattern, text)
    
    if image_match:
        # Extract and process image
        image_path = image_match.group(1)
        print(f"检测到图像引用: {image_path}")
        
        # Remove image reference from text
        text = re.sub(image_pattern, '', text).strip()
        
        # Process image if it exists
        if os.path.exists(image_path):
            image_description = await multimodal_system.process_image(image_path)
            print(f"图像分析: {image_description}")
        else:
            print(f"警告: 图像文件不存在: {image_path}")
    
    # Extract emotional tone
    user_emotion = await emotion_system.analyze_user_emotion(text)
    print(f"检测到情绪: {user_emotion['emotion']} (强度: {user_emotion['intensity']})")
    
    # Get contextual information
    context_data = await context_system.get_current_context()
    
    # Extract potential preferences
    preferences = await preference_tracker.extract_preferences(text)
    if preferences:
        print("检测到新的偏好信息:")
        for category, prefs in preferences.items():
            for key, value in prefs.items():
                print(f"- {category}.{key}: {value}")
    
    # 尝试提取日历事件（在生成回复前）
    calendar_system = CalendarSystem()
    await calendar_system.initialize()
    event_result = await extract_calendar_events_from_text(text, calendar_system)
    
    # Get AI response with all context
    response = await get_enhanced_ai_response(
        text, 
        user_emotion=user_emotion,
        context_data=context_data
    )
    
    # 如果提取到日历事件，添加提示
    if event_result.get("added", False):
        event_added_message = f"\n\n已将事件「{event_result['title']}」添加到您的日历，时间是{event_result['date_time']}。"
        response += event_added_message
    
    print(f"AI回应: {response}")
    
    # Generate speech with appropriate emotion
    ai_emotion = await emotion_system.detect_ai_emotion(response)
    generate_speech_with_emotion(response, emotion=ai_emotion)
    
    print('-' * 50)

async def get_enhanced_ai_response(text, user_emotion=None, context_data=None):
    """Enhanced version of get_ai_response with all new context features"""
    print("AI正在处理...")
    
    try:
        # Check if browser search is needed based on (y) marker
        use_browser = await should_use_browser(text)
        
        browser_result = ""
        if use_browser:
            try:
                browser_result = await run_browser_search(text)
                print(f"浏览器搜索结果:\n{browser_result}\n{'='*50}")
            except Exception as browser_e:
                print(f"浏览器搜索失败: {browser_e}")
        
        # Construct query - remove (y)/(n) markers before processing
        clean_text = re.sub(r'\([yn]\)', '', text).strip()
        
        # Build enhanced context
        enhanced_context = ""
        
        # Add emotional context
        if user_emotion:
            enhanced_context += f"用户当前情绪: {user_emotion['emotion']} (强度: {user_emotion['intensity']})\n\n"
        
        # Add contextual awareness
        if context_data:
            enhanced_context += "当前环境信息:\n"
            for key, value in context_data.items():
                enhanced_context += f"- {key}: {value}\n"
            enhanced_context += "\n"
        
        # Add browser results if any
        if browser_result:
            enhanced_context += f"网络搜索结果:\n{browser_result}\n\n"
            
        # 获取用户即将到来的日历事件
        calendar_system = CalendarSystem()
        await calendar_system.initialize()
        
        upcoming_events = await calendar_system.get_upcoming_events(days=3)
        if upcoming_events:
            enhanced_context += "即将到来的日历事件:\n"
            for event in upcoming_events[:3]:  # 最多显示3个
                enhanced_context += f"- {event['title']} ({event['time_desc']})\n"
            enhanced_context += "\n"
        
        # Get current AI mood
        current_mood = await get_current_mood()
        if current_mood != "neutral":
            enhanced_context += f"AI当前心情: {current_mood}\n\n"
        
        # Construct final query with all context
        if enhanced_context:
            query = f"""
            问题: {clean_text}
            
            上下文信息:
            {enhanced_context}
            
            请基于这些信息以01女友的身份提供有帮助、准确、对话式的回应。
            回应应该与当前检测到的用户情绪相匹配，并考虑上述所有情境因素。
            """
        else:
            query = clean_text
        
        # Get RAG response
        response = await rag_system.get_answer(query)
        
        # Update user preferences based on interaction
        asyncio.create_task(update_preferences_from_interaction(text, response))
        
        return response
    
    except Exception as e:
        print(f"AI响应生成中出错: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        if "ConnectionError" in str(e) or "Connection refused" in str(e):
            return "抱歉，无法连接到服务。请确认服务已启动且端口可访问。"
        
        return "抱歉，我现在无法回应。"

async def shutdown(tasks, threads):
    """Handle graceful shutdown"""
    global exit_flag
    
    print("\n正在关闭系统...")
    exit_flag = True
    
    try:
        # Cancel all async tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save all data before exit
        await consolidate_memories()
        
        # Wait for threads (they should check exit_flag and terminate)
        timeout = 3  # Maximum wait time in seconds
        start_time = time.time()
        active_threads = True
        
        while active_threads and (time.time() - start_time) < timeout:
            active_threads = False
            for t in threading.enumerate():
                if t != threading.current_thread() and t.is_alive() and not t.daemon:
                    active_threads = True
            if active_threads:
                await asyncio.sleep(0.2)
        
        print("系统已关闭。")
    except Exception as e:
        print(f"关闭过程中出错: {e}")
        print(traceback.format_exc())

async def save_all_data(preference_tracker):
    """Save all persistent data"""
    try:
        # Consolidate memories
        await consolidate_memories()
        
        # Save preferences
        await preference_tracker.save()
        
        # Save mood state
        await save_mood_state()
        
        # Save voice style preference
        await save_voice_preferences()
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def print_help_menu():
    """Print the help menu with all available commands"""
    print("\n=== 可用命令 ===")
    print("  help                - 显示此帮助信息")
    print("  quit                - 保存并退出程序")
    print("\n=== 语音控制 ===")
    print("  voice on/off        - 启用/禁用语音识别模式")
    print("  voice_style list    - 列出可用的语音风格")
    print("  voice_style set <style> - 设置语音风格")
    print("\n=== 交互模式 ===")
    print("  random_chat on/off  - 启用/禁用随机对话启动")
    print("  mood status         - 显示当前AI心情")
    print("  mood set <mood>     - 设置AI心情(happy/sad/neutral/excited)")
    print("  sleep on/off        - 启用/禁用休眠模式")
    print(f"  {WAKE_PHRASE}           - 唤醒AI")
    print("\n=== 知识与记忆 ===")
    print("  rag rebuild         - 重建RAG知识库索引")
    print("  memory trim         - 手动优化记忆容量")
    print("  memory stats        - 显示记忆统计信息")
    print("  learn <topic>       - 主动学习特定主题")
    print("  learn               - 从最近对话中学习重要主题")
    print("\n=== 用户偏好 ===")
    print("  preference show     - 显示已记录的用户偏好")
    print("  preference set <category> <key> <value> - 设置用户偏好")
    print("\n=== 日历功能 ===")
    print('  calendar add "标题" "YYYY-MM-DD HH:MM" [描述] - 添加日历事件')
    print("  calendar list [天数] - 列出未来事件")
    print("  calendar today      - 显示今天的事件")
    print("  calendar delete <ID> - 删除事件")
    print("\n=== 其他功能 ===")
    print("  summary             - 生成今日对话总结")
    print("  suggest             - 获取活动建议")
    print("  context             - 显示当前环境信息(时间/日期/天气)")
    print("  image capture       - 捕获并分析图像")
    print("  save                - 手动保存当前所有数据")
    print("\n任何其他输入将被视为对话消息发送给AI")

# Utility function for asynchronous input
async def async_input(prompt):
    """Get input asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

# Preference Tracker
class PreferenceTracker:
    """System for tracking and managing user preferences"""
    
    def __init__(self, filename=PREFERENCE_FILE):
        self.filename = filename
        self.preferences = {
            "general": {},
            "food": {},
            "entertainment": {},
            "schedule": {},
            "personal": {}
        }
    
    async def load(self):
        """Load preferences from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.preferences.update(data)
            print("用户偏好加载完成")
        except Exception as e:
            print(f"加载用户偏好时出错: {e}")
    
    async def save(self):
        """Save preferences to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            print("用户偏好已保存")
        except Exception as e:
            print(f"保存用户偏好时出错: {e}")
    
    async def set_preference(self, category, key, value):
        """Set a specific preference"""
        if category not in self.preferences:
            self.preferences[category] = {}
        
        self.preferences[category][key] = value
        await self.save()
    
    async def get_preference(self, category, key, default=None):
        """Get a specific preference"""
        if category in self.preferences and key in self.preferences[category]:
            return self.preferences[category][key]
        return default
    
    async def get_all_preferences(self):
        """Get all preferences"""
        return self.preferences
    
    async def extract_preferences(self, text):
        """Extract potential preferences from text"""
        try:
            # Simple extraction using keywords
            extracted = {}
            
            # Food preferences
            food_patterns = [
                (r'喜欢吃(.*?)(?:，|。|！|\s|$)', 'food', 'likes'),
                (r'不喜欢吃(.*?)(?:，|。|！|\s|$)', 'food', 'dislikes'),
                (r'最爱的食物是(.*?)(?:，|。|！|\s|$)', 'food', 'favorite')
            ]
            
            # Entertainment preferences
            entertainment_patterns = [
                (r'喜欢看(.*?)(?:，|。|！|\s|$)', 'entertainment', 'likes'),
                (r'最喜欢的电影是(.*?)(?:，|。|！|\s|$)', 'entertainment', 'favorite_movie'),
                (r'最喜欢的游戏是(.*?)(?:，|。|！|\s|$)', 'entertainment', 'favorite_game')
            ]
            
            # Schedule preferences
            schedule_patterns = [
                (r'通常(.*?)点起床', 'schedule', 'wake_time'),
                (r'通常(.*?)点睡觉', 'schedule', 'sleep_time'),
                (r'周末喜欢(.*?)(?:，|。|！|\s|$)', 'schedule', 'weekend_activity')
            ]
            
            # Personal preferences
            personal_patterns = [
                (r'最喜欢的颜色是(.*?)(?:，|。|！|\s|$)', 'personal', 'favorite_color'),
                (r'我的生日是(.*?)(?:，|。|！|\s|$)', 'personal', 'birthday')
            ]
            
            all_patterns = food_patterns + entertainment_patterns + schedule_patterns + personal_patterns
            
            for pattern, category, key in all_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    if category not in extracted:
                        extracted[category] = {}
                    extracted[category][key] = matches[0].strip()
                    # Save preference directly
                    await self.set_preference(category, key, matches[0].strip())
            
            # For more complex preference extraction that requires AI understanding
            if len(text) > 20 and not extracted:  # Only for longer texts and if nothing extracted yet
                try:
                    prompt = f"""
                    分析以下文本，提取出可能的用户偏好:
                    "{text}"
                    
                    如果检测到偏好，请以JSON格式返回，格式如下:
                    {{
                      "category": "分类",
                      "key": "偏好名",
                      "value": "偏好值"
                    }}
                    
                    分类可以是: food(食物), entertainment(娱乐), schedule(日程), personal(个人), general(一般)
                    如果没有检测到任何偏好，返回 {{"found": false}}
                    """
                    
                    response = await client.chat.completions.create(
                        model=OPENROUTER_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    
                    if "found" not in result or result["found"] != False:
                        if "category" in result and "key" in result and "value" in result:
                            category = result["category"]
                            key = result["key"]
                            value = result["value"]
                            
                            if category not in extracted:
                                extracted[category] = {}
                            extracted[category][key] = value
                            
                            # Save preference directly
                            await self.set_preference(category, key, value)
                
                except Exception as e:
                    print(f"AI偏好提取时出错: {e}")
            
            return extracted
        
        except Exception as e:
            print(f"提取偏好时出错: {e}")
            return {}

# Function to update preferences based on conversation
async def update_preferences_from_interaction(user_text, ai_response):
    """Extract and update user preferences from conversation"""
    try:
        # This is a placeholder for the actual implementation
        # In a real implementation, you would analyze both the user's text
        # and the AI's response to extract preferences
        # For example, if the AI asks "What's your favorite food?" and
        # the user responds "I love pizza", you would extract that preference
        
        pass  # Will be implemented in a future update
    except Exception as e:
        print(f"更新用户偏好时出错: {e}")

# Emotional Intelligence System
class EmotionalIntelligenceSystem:
    """System for detecting and responding to emotions in conversation"""
    
    def __init__(self):
        self.emotions_file = "emotions_data.json"
        self.emotion_history = []
        self.emotion_cache = {}
    
    async def initialize(self):
        """Initialize the emotion system"""
        try:
            if os.path.exists(self.emotions_file):
                with open(self.emotions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.emotion_history = data.get("history", [])
                    self.emotion_cache = data.get("cache", {})
            print("情感系统初始化完成")
        except Exception as e:
            print(f"初始化情感系统时出错: {e}")
            self.emotion_history = []
            self.emotion_cache = {}
    
    async def analyze_user_emotion(self, text):
        """Analyze user's emotional state from text"""
        try:
            # Check cache for similar inputs to avoid repeated API calls
            cache_key = text[:50]  # Use first 50 chars as key
            if cache_key in self.emotion_cache:
                return self.emotion_cache[cache_key]
            
            emotion_prompt = f"分析以下消息中的情感色彩: '{text}'. 识别主要情绪(开心、悲伤、愤怒、中性等)及其强度(低、中、高)。仅返回JSON格式: {{\"emotion\": \"情绪名称\", \"intensity\": \"强度\"}}"
            
            # Get response from OpenRouter
            response = await client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": emotion_prompt}],
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure we have valid fields
            if "emotion" not in result or "intensity" not in result:
                result = {"emotion": "neutral", "intensity": "medium"}
            
            # Cache the result
            self.emotion_cache[cache_key] = result
            
            # Add to history with timestamp
            self.emotion_history.append({
                "text": text,
                "emotion": result["emotion"],
                "intensity": result["intensity"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Trim history if it gets too long
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-50:]
            
            # Save periodically
            if len(self.emotion_history) % 10 == 0:
                asyncio.create_task(self.save())
                
            return result
            
        except Exception as e:
            print(f"分析情绪时出错: {e}")
            return {"emotion": "neutral", "intensity": "medium"}
    
    async def detect_ai_emotion(self, text):
        """Detect which emotion should be expressed in AI's response"""
        try:
            # Simple keyword matching for efficiency
            emotion_keywords = {
                "happy": ["高兴", "开心", "快乐", "欣喜", "喜欢", "好开心", "真棒", "太好了", "爱你", "哈哈"],
                "sad": ["难过", "伤心", "悲伤", "遗憾", "可惜", "伤感", "哭", "不开心", "失望"],
                "excited": ["激动", "兴奋", "惊喜", "太棒了", "amazing", "震惊", "哇", "惊讶", "期待"],
                "angry": ["生气", "愤怒", "讨厌", "烦", "烦躁", "不爽", "火大"]
            }
            
            # Default emotion
            default_emotion = "neutral"
            
            # Check for emotion keywords
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        return emotion
            
            return default_emotion
        
        except Exception as e:
            print(f"检测AI情绪时出错: {e}")
            return "neutral"
    
    async def save(self):
        """Save emotion data to file"""
        try:
            with open(self.emotions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "history": self.emotion_history,
                    "cache": self.emotion_cache
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存情感数据时出错: {e}")

# Multimodal System for image processing
class MultimodalSystem:
    """System for handling multimodal inputs like images"""
    
    def __init__(self):
        self.image_dir = "recorded_images"
        self.processed_images = {}
    
    async def initialize(self):
        """Initialize the multimodal system"""
        os.makedirs(self.image_dir, exist_ok=True)
        print("多模态系统初始化完成")
    
    async def process_image(self, image_path):
        """Process an image and return a description"""
        try:
            if image_path in self.processed_images:
                return self.processed_images[image_path]
                
            # For OpenRouter models with vision capabilities
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = await client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "用中文详细描述这张图片中的内容，注意捕捉情感、环境和关键细节。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
            )
            
            description = response.choices[0].message.content
            
            # Cache the result
            self.processed_images[image_path] = description
            
            # Also log this as conversation context
            image_exchange = {
                "user": f"[用户分享了一张图片: {os.path.basename(image_path)}]",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": 2
            }
            
            asyncio.create_task(async_save_short_term_memory(image_exchange))
            
            # Generate a response about the image
            response_text = f"我看到了这张图片。{description}"
            print(f"图片描述: {description}")
            generate_speech(response_text)
            
            return description
            
        except Exception as e:
            print(f"处理图片时出错: {e}")
            error_msg = "抱歉，我无法清晰地看到这张图片。"
            generate_speech(error_msg)
            return error_msg

# Contextual Awareness System
class ContextualAwarenessSystem:
    """System for tracking contextual information like time and environment"""
    
    def __init__(self):
        self.context = {}
        self.weather_api_key = None  # Add your API key if you want weather info
        self.location = "Beijing"  # Default location
        self.update_interval = 300  # Update every 5 minutes
    
    async def get_current_context(self):
        """Get current contextual information"""
        if not self.context or (time.time() - self.context.get("last_update", 0)) > 60:
            await self.update_context()
        return self.context
    
    async def update_context(self):
        """Update contextual information"""
        try:
            now = datetime.now()
            
            self.context = {
                "time": now.strftime("%H:%M"),
                "date": now.strftime("%Y-%m-%d"),
                "day_of_week": now.strftime("%A"),
                "part_of_day": self.get_part_of_day(now.hour),
                "last_update": time.time()
            }
            
            # Add weather if API key is available
            if self.weather_api_key:
                try:
                    weather = await self.get_weather()
                    if weather:
                        self.context.update(weather)
                except Exception as weather_e:
                    print(f"获取天气信息时出错: {weather_e}")
            
            return self.context
            
        except Exception as e:
            print(f"更新上下文信息时出错: {e}")
            return {
                "time": datetime.now().strftime("%H:%M"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "last_update": time.time()
            }
    
    def get_part_of_day(self, hour):
        """Determine part of day based on hour"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    async def get_weather(self):
        """Get weather information from API"""
        if not self.weather_api_key:
            return {}
            
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={self.weather_api_key}&units=metric"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "weather": data["weather"][0]["description"],
                            "temperature": f"{data['main']['temp']}°C",
                            "humidity": f"{data['main']['humidity']}%",
                            "wind_speed": f"{data['wind']['speed']} m/s"
                        }
                    else:
                        return {}
        except Exception:
            return {}
    
    async def update_loop(self):
        """Loop to periodically update contextual information"""
        global exit_flag
        while not exit_flag:
            await self.update_context()
            await asyncio.sleep(self.update_interval)

# Helper function to get current AI mood
async def get_current_mood():
    """Get the current AI mood"""
    global current_ai_mood
    return current_ai_mood

# Helper function to set AI mood
async def set_ai_mood(mood):
    """Set the AI mood"""
    global current_ai_mood
    current_ai_mood = mood
    await save_mood_state()
    return current_ai_mood

# Save and load mood state
async def save_mood_state():
    """Save current mood state to file"""
    try:
        with open(MOOD_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"mood": current_ai_mood, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
    except Exception as e:
        print(f"保存心情状态时出错: {e}")

async def load_mood_state():
    """Load mood state from file"""
    global current_ai_mood
    try:
        if os.path.exists(MOOD_STATE_FILE):
            with open(MOOD_STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_ai_mood = data.get("mood", "neutral")
    except Exception as e:
        print(f"加载心情状态时出错: {e}")
        current_ai_mood = "neutral"

# Function to generate speech with emotional tone
def generate_speech_with_emotion(text, emotion="neutral"):
    """Generate speech with emotional tone"""
    # Map emotion to voice style parameter
    emotion_to_style = {
        "happy": "happy",
        "sad": "sad",
        "excited": "excited",
        "angry": "angry",
        "neutral": "normal"
    }
    
    # Get voice style from emotion or use current style
    style = emotion_to_style.get(emotion, current_voice_style)
    
    # Clean text for speech by removing any importance markers
    clean_text = re.sub(r'\((?:高|中|低)\)', '', text).strip()
    
    # Base parameters
    params = {
        "refer_wav_path": REFERENCE_WAV,
        "prompt_text": "可聪明的人从一开始就不会入局。你瞧，我是不是更聪明一点？",
        "prompt_language": "zh",
        "text": clean_text,
        "text_language": "zh",
        "style": style  # Add style parameter
    }

    try:
        # Note: For this to work, your TTS service needs to support the style parameter
        # If it doesn't, you'll need to modify this code to match your TTS API
        response = requests.get('http://localhost:9880/', params=params)
        response.raise_for_status()

        file_path = 'temp.wav'
        with open(file_path, 'wb') as f:
            f.write(response.content)

        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait()
        
    except requests.exceptions.RequestException as e:
        print(f"请求TTS服务时出错: {e}")
        # Fallback to regular speech generation
        generate_speech(text)
    except Exception as e:
        print(f"情感语音生成期间出错: {e}")
        # Fallback to regular speech generation
        generate_speech(text)

# Helper functions for voice style management
async def change_voice_style(style):
    """Change the voice style"""
    global current_voice_style
    if style in voice_styles:
        current_voice_style = style
        await save_voice_preferences()
        return True
    return False

async def list_voice_styles():
    """List available voice styles"""
    return voice_styles

async def save_voice_preferences():
    """Save voice preferences to file"""
    try:
        with open(VOICE_PREF_FILE, 'w', encoding='utf-8') as f:
            json.dump({"style": current_voice_style}, f)
    except Exception as e:
        print(f"保存语音偏好时出错: {e}")

async def load_voice_preferences():
    """Load voice preferences from file"""
    global current_voice_style
    try:
        if os.path.exists(VOICE_PREF_FILE):
            with open(VOICE_PREF_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_voice_style = data.get("style", "normal")
    except Exception as e:
        print(f"加载语音偏好时出错: {e}")
        current_voice_style = "normal"

# Functions for mood evolution and random shifts
async def mood_evolution_loop():
    """Loop to gradually evolve the AI's mood over time"""
    global exit_flag
    global current_ai_mood
    
    print("情绪演化线程已启动")
    
    # Possible mood transitions
    mood_transitions = {
        "neutral": ["happy", "sad", "neutral"],
        "happy": ["neutral", "excited", "happy"],
        "sad": ["neutral", "sad"],
        "excited": ["happy", "neutral", "excited"],
        "angry": ["neutral", "sad", "angry"]
    }
    
    # Different intervals for updates based on current mood
    # Some moods change faster than others
    mood_intervals = {
        "neutral": 3600,    # 1 hour
        "happy": 1800,      # 30 minutes
        "sad": 2700,        # 45 minutes 
        "excited": 1200,    # 20 minutes
        "angry": 900        # 15 minutes
    }
    
    while not exit_flag:
        try:
            # Get current mood and its transition interval
            current_mood = await get_current_mood()
            interval = mood_intervals.get(current_mood, 3600)
            
            # Wait for the mood transition interval (or until exit_flag is set)
            for _ in range(interval):
                if exit_flag:
                    break
                await asyncio.sleep(1)
            
            if exit_flag:
                break
                
            # Small chance to change mood
            if random.random() < 0.3:  # 30% chance to change mood
                # Get possible transitions for current mood
                possible_moods = mood_transitions.get(current_mood, ["neutral"])
                # Choose a new mood
                new_mood = random.choice(possible_moods)
                # Set the new mood if it's different
                if new_mood != current_mood:
                    print(f"AI心情从 {current_mood} 变为 {new_mood}")
                    await set_ai_mood(new_mood)
        
        except Exception as e:
            print(f"情绪演化过程中出错: {e}")
            await asyncio.sleep(60)  # Wait before retrying

# Function to capture images (if camera is available)
async def capture_image():
    """Capture an image using the camera if available"""
    try:
        # Ensure the directory exists
        os.makedirs("recorded_images", exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"recorded_images/image_{timestamp}.jpg"
        
        # Use ffmpeg to capture from camera (works on most platforms)
        subprocess.run([
            "ffmpeg",
            "-f", "dshow" if os.name == 'nt' else "avfoundation",
            "-i", "video=Webcam" if os.name == 'nt' else "0",
            "-frames:v", "1",
            "-y",
            image_path
        ], check=True, capture_output=True)
        
        if os.path.exists(image_path):
            return image_path
        return None
    except Exception as e:
        print(f"捕获图像时出错: {e}")
        return None

# Command Parser
class CommandParser:
    """Parse user commands"""
    
    def __init__(self):
        """Initialize command parser with command patterns"""
        self.command_patterns = {
            r'^help$': {"type": "help"},
            r'^quit$|^exit$': {"type": "quit"},
            r'^voice\s+(on|off)$': {"type": "voice", "args": {"state": "{1}"}},
            r'^random_chat\s+(on|off)$': {"type": "random_chat", "args": {"state": "{1}"}},
            r'^rag\s+rebuild$': {"type": "rag", "args": {"action": "rebuild"}},
            r'^save$': {"type": "save"},
            r'^memory\s+trim$': {"type": "memory", "args": {"action": "trim"}},
            r'^memory\s+stats$': {"type": "memory", "args": {"action": "stats"}},
            r'^summary$': {"type": "summary"},
            r'^suggest$': {"type": "suggest"},
            r'^mood\s+set\s+(\w+)$': {"type": "mood", "args": {"action": "set", "value": "{1}"}},
            r'^mood\s+status$': {"type": "mood", "args": {"action": "status"}},
            r'^voice_style\s+list$': {"type": "voice_style"},
            r'^voice_style\s+set\s+(\w+)$': {"type": "voice_style", "args": {"style": "{1}"}},
            r'^context$': {"type": "context"},
            r'^learn\s+(.+)$': {"type": "learn", "args": {"topic": "{1}"}},
            r'^learn$': {"type": "learn"},
            r'^preference\s+show$': {"type": "preference", "args": {"action": "show"}},
            r'^preference\s+set\s+(\w+)\s+(\w+)\s+(.+)$': {
                "type": "preference", 
                "args": {"action": "set", "category": "{1}", "key": "{2}", "value": "{3}"}
            },
            r'^image\s+capture$': {"type": "image", "args": {"action": "capture"}},
            r'^calendar\s+add\s+"([^"]+)"\s+"([^"]+)"\s*(.*)$': {
                "type": "calendar", 
                "args": {"action": "add", "title": "{1}", "datetime": "{2}", "desc": "{3}"}
            },
            r'^calendar\s+list\s*(\d*)$': {
                "type": "calendar", 
                "args": {"action": "list", "days": "{1}"}
            },
            r'^calendar\s+today$': {"type": "calendar", "args": {"action": "today"}},
            r'^calendar\s+delete\s+(\w+)$': {
                "type": "calendar", 
                "args": {"action": "delete", "id": "{1}"}
            },
            r'^sleep\s+(on|off)$': {"type": "sleep", "args": {"state": "{1}"}},
            r'^' + WAKE_PHRASE + '$': {"type": "wake"} # 添加唤醒词命令
        }
    
    def is_command(self, text):
        """Check if text is a command"""
        for pattern in self.command_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def parse_command(self, text):
        """Parse a command string into command type and arguments"""
        for pattern, command_info in self.command_patterns.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Deep copy command info to avoid modifying the original
                cmd_type = command_info["type"]
                args = command_info.get("args", {}).copy()
                
                # Replace placeholders with actual values
                for arg_name, arg_value in args.items():
                    if isinstance(arg_value, str) and arg_value.startswith("{") and arg_value.endswith("}"):
                        group_num = int(arg_value[1:-1])
                        if group_num <= len(match.groups()):
                            args[arg_name] = match.group(group_num)
                
                return cmd_type, args
        
        # Default return if no match
        return "unknown", {}

async def initiate_shutdown(tasks):
    """Prepare for system shutdown"""
    global exit_flag
    
    print("正在准备关闭系统...")
    exit_flag = True
    
    # Save all data
    await consolidate_memories()
    print("记忆已整合")
    
    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()
    
    # Allow tasks to complete cancellation
    await asyncio.sleep(2)
    
    print("系统已安全关闭。")

async def process_image(image_path):
    """Process an image outside the multimodal system context"""
    try:
        print(f"处理图像: {image_path}")
        
        # Create a temporary multimodal system for this operation
        temp_multimodal = MultimodalSystem()
        await temp_multimodal.initialize()
        
        # Process the image
        description = await temp_multimodal.process_image(image_path)
        return description
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return "无法处理图像"

# Helper functions for memory management
async def trim_all_memories():
    """Trim both short and long term memories"""
    try:
        # Trim long-term memory
        long_term = load_chat_history()
        if long_term:
            trimmed_long_term = await trim_memory(long_term)
            with open(LONG_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
                json.dump(trimmed_long_term, f, ensure_ascii=False, indent=2)
            print(f"长期记忆优化: {len(long_term)} -> {len(trimmed_long_term)}")
            
        # Trim short-term memory
        short_term = await load_short_term_memory()
        if short_term:
            trimmed_short_term = await trim_memory(short_term, max_size=100)
            with open(SHORT_TERM_MEM_FILE, 'w', encoding='utf-8') as f:
                json.dump(trimmed_short_term, f, ensure_ascii=False, indent=2)
            print(f"短期记忆优化: {len(short_term)} -> {len(trimmed_short_term)}")
            
        return True
    except Exception as e:
        print(f"优化记忆时出错: {e}")
        return False

async def show_memory_stats():
    """Show memory statistics"""
    try:
        # Get memory counts
        long_term = load_chat_history()
        short_term = await load_short_term_memory()
        
        print("\n=== 记忆统计 ===")
        print(f"长期记忆: {len(long_term)} 条")
        print(f"短期记忆: {len(short_term)} 条")
        print(f"总记忆量: {len(long_term) + len(short_term)} 条")
        
        # Count by priority
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for memory in long_term + short_term:
            priority = memory.get("priority", 2)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print("\n按重要性统计:")
        print(f"低重要性(1): {priority_counts.get(1, 0)} 条")
        print(f"中等重要性(2): {priority_counts.get(2, 0)} 条")
        print(f"高重要性(3): {priority_counts.get(3, 0)} 条")
        print(f"永久记忆(4): {priority_counts.get(4, 0)} 条")
        
        # Count by time periods
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        time_counts = {"today": 0, "yesterday": 0, "week": 0, "month": 0, "older": 0}
        
        for memory in long_term + short_term:
            try:
                timestamp_str = memory.get("timestamp", "")
                if not timestamp_str:
                    time_counts["older"] += 1
                    continue
                    
                memory_time = datetime.fromisoformat(timestamp_str)
                
                if memory_time >= today:
                    time_counts["today"] += 1
                elif memory_time >= yesterday:
                    time_counts["yesterday"] += 1
                elif memory_time >= week_ago:
                    time_counts["week"] += 1
                elif memory_time >= month_ago:
                    time_counts["month"] += 1
                else:
                    time_counts["older"] += 1
            except:
                time_counts["older"] += 1
        
        print("\n按时间统计:")
        print(f"今天: {time_counts['today']} 条")
        print(f"昨天: {time_counts['yesterday']} 条")
        print(f"过去一周: {time_counts['week']} 条")
        print(f"过去一月: {time_counts['month']} 条")
        print(f"更早: {time_counts['older']} 条")
        
    except Exception as e:
        print(f"获取记忆统计时出错: {e}")

# Proactive Learning System
async def learn_topic(topic):
    """Learn about a specific topic using the web and add to knowledge base"""
    try:
        print(f"学习关于 '{topic}' 的信息...")
        
        # Get search results about the topic
        browser_result = await run_browser_search(topic)
        
        if not browser_result:
            print("无法获取相关信息")
            return False
        
        # Format the knowledge
        formatted_knowledge = f"""
        主题: {topic}
        获取时间: {datetime.now(timezone.utc).isoformat()}
        来源: 网络搜索
        
        内容:
        {browser_result}
        """
        
        # Save to knowledge base
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
        # Create a safe filename
        safe_filename = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        knowledge_file = os.path.join(KNOWLEDGE_DIR, f"{safe_filename}_{datetime.now().strftime('%Y%m%d')}.txt")
        
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            f.write(formatted_knowledge)
        
        print(f"已将知识保存到: {knowledge_file}")
        
        # Summarize what was learned
        summary_prompt = f"简要总结以下关于'{topic}'的信息，不超过100字:\n{browser_result}"
        
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150
        )
        
        summary = response.choices[0].message.content
        print(f"学习总结: {summary}")
        
        # Add to conversation as AI message
        ai_message = f"我刚刚学习了关于'{topic}'的信息。{summary}"
        bot_exchange = {"bot": ai_message}
        asyncio.create_task(async_save_short_term_memory(bot_exchange))
        
        # Also say it out loud
        generate_speech(ai_message)
        
        return True
        
    except Exception as e:
        print(f"学习主题时出错: {e}")
        traceback.print_exc()
        return False

async def learn_from_interactions():
    """Analyze recent interactions to identify topics to learn about"""
    try:
        print("分析最近对话...")
        
        # Load recent interactions
        short_term = await load_short_term_memory()
        
        # Extract text from interactions
        interactions_text = ""
        for item in short_term[-20:]:  # Use last 20 interactions
            if "user" in item:
                interactions_text += f"用户: {item['user']}\n"
            if "bot" in item:
                interactions_text += f"AI: {item['bot']}\n"
        
        if not interactions_text:
            print("没有足够的对话用于分析")
            return
        
        # Ask model to identify interesting topics
        topic_prompt = f"""
        分析以下对话，识别出需要深入学习的3个主题。这些主题应该是:
        1. 用户表现出兴趣的
        2. 需要更多背景知识的
        3. 可以通过网络搜索获取信息的
        
        对话内容:
        {interactions_text}
        
        请列出3个最值得学习的主题，格式为JSON数组:
        [
          "主题1",
          "主题2",
          "主题3"
        ]
        """
        
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": topic_prompt}],
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        try:
            topics = json.loads(response.choices[0].message.content)
            
            # Ensure we have a list
            if not isinstance(topics, list):
                if isinstance(topics, dict) and "topics" in topics:
                    topics = topics["topics"]
                else:
                    topics = []
            
            if topics:
                print(f"识别出 {len(topics)} 个学习主题:")
                for topic in topics:
                    print(f"- {topic}")
                    # Learn about each topic
                    await learn_topic(topic)
            else:
                print("未识别出需要学习的主题")
                
        except json.JSONDecodeError:
            print("无法解析主题列表")
        
    except Exception as e:
        print(f"从对话中学习时出错: {e}")
        traceback.print_exc()

async def suggest_activity(context_system, preference_tracker):
    """Suggest an activity based on context and preferences"""
    try:
        # Get current context
        context = await context_system.get_current_context()
        
        # Get user preferences
        preferences = await preference_tracker.get_all_preferences()
        
        # Format preferences for prompt
        preferences_text = ""
        for category, prefs in preferences.items():
            if prefs:
                preferences_text += f"\n{category}偏好:\n"
                for key, value in prefs.items():
                    preferences_text += f"- {key}: {value}\n"
        
        # Create suggestion prompt
        suggestion_prompt = f"""
        基于以下信息，为用户推荐一个有趣的活动:
        
        当前时间: {context.get('time', 'unknown')}
        当前日期: {context.get('date', 'unknown')}
        星期几: {context.get('day_of_week', 'unknown')}
        一天中的时段: {context.get('part_of_day', 'unknown')}
        
        用户偏好:
        {preferences_text if preferences_text else "尚无记录的偏好信息"}
        
        请提供一个温暖、个性化的活动建议，以第一人称回复，像女朋友会给男朋友的建议。
        回复应简洁、体贴，不超过75字。
        """
        
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": suggestion_prompt}],
            max_tokens=250
        )
        
        suggestion = response.choices[0].message.content
        
        # Save suggestion to memory
        suggestion_exchange = {"bot": suggestion}
        asyncio.create_task(async_save_short_term_memory(suggestion_exchange))
        
        return suggestion
        
    except Exception as e:
        print(f"生成活动建议时出错: {e}")
        traceback.print_exc()
        return "抱歉，我现在无法为你提供活动建议。"

async def proactive_learning_loop():
    """Background loop for proactive learning"""
    global exit_flag
    
    print("主动学习线程已启动")
    check_interval = 1800  # 30 minutes
    
    while not exit_flag:
        try:
            # Wait for interval
            for _ in range(check_interval):
                if exit_flag:
                    break
                await asyncio.sleep(1)
            
            if exit_flag:
                break
            
            # Only learn if there's been activity
            short_term = await load_short_term_memory()
            if short_term and len(short_term) > 5:  # Only if we have some conversation
                print("执行定期主动学习...")
                await learn_from_interactions()
                
        except Exception as e:
            print(f"主动学习过程中出错: {e}")
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait before retrying

async def daily_summary_scheduler():
    """Schedule daily summaries"""
    global exit_flag
    
    print("每日总结调度器已启动")
    
    while not exit_flag:
        try:
            # Get current time
            now = datetime.now()
            
            # Calculate time until next summary (9:00 PM)
            target_hour = 21
            target_minute = 0
            
            if now.hour < target_hour or (now.hour == target_hour and now.minute < target_minute):
                # Today's target time
                target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            else:
                # Tomorrow's target time
                target = (now + timedelta(days=1)).replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            
            # Calculate seconds until target
            seconds_until_target = (target - now).total_seconds()
            
            print(f"下一次总结将在 {target.strftime('%Y-%m-%d %H:%M')} 生成，还有 {seconds_until_target:.1f} 秒")
            
            # Wait until target time (checking exit_flag every second)
            for _ in range(int(seconds_until_target)):
                if exit_flag:
                    break
                await asyncio.sleep(1)
            
            if exit_flag:
                break
            
            # Generate and announce summary
            summary = await generate_daily_summary()
            print(f"\n今日总结: {summary}")
            generate_speech(summary)
            
        except Exception as e:
            print(f"每日总结调度器出错: {e}")
            traceback.print_exc()
            await asyncio.sleep(3600)  # Retry in an hour

async def generate_daily_summary():
    """Generate a summary of today's conversations"""
    try:
        print("生成今日对话总结...")
        
        # Load today's conversations
        short_term = await load_short_term_memory()
        long_term = load_chat_history()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Filter for today's conversations
        today_conversations = []
        
        for memory in short_term + long_term:
            try:
                timestamp = memory.get("timestamp", "")
                if timestamp and today in timestamp:
                    today_conversations.append(memory)
            except:
                continue
        
        if not today_conversations:
            return "今天我们还没有进行过对话。"
        
        # Extract text from conversations
        conversation_text = ""
        for item in today_conversations:
            if "user" in item:
                conversation_text += f"用户: {item['user']}\n"
            if "bot" in item:
                conversation_text += f"AI: {item['bot']}\n"
        
        # Generate summary
        summary_prompt = f"""
        总结以下今日的对话，包括:
        1. 主要讨论的话题
        2. 重要的决定或计划
        3. 显著的情感交流
        4. 有趣的时刻
        
        使用第一人称"我"来描述AI，使用"你"来描述用户。
        总结应该温暖友好，像女朋友总结一天的对话。
        总结不超过200字。
        
        对话内容:
        {conversation_text[:2000]}  # Limit to avoid token limits
        """
        
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        
        # Save summary to memory with high importance
        summary_exchange = {"bot": f"(高) 今日总结: {summary}"}
        asyncio.create_task(async_save_short_term_memory(summary_exchange))
        
        return summary
        
    except Exception as e:
        print(f"生成今日总结时出错: {e}")
        traceback.print_exc()
        return "抱歉，我无法生成今日总结。"

# Calendar System
class CalendarSystem:
    """系统用于管理日历事件和提醒"""
    
    def __init__(self, filename=CALENDAR_FILE):
        self.filename = filename
        self.events = []
        self.reminders_active = True
    
    async def initialize(self):
        """初始化日历系统"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.events = json.load(f)
            print("日历系统初始化完成")
            
            # 清理过期事件
            await self.clean_past_events()
        except Exception as e:
            print(f"初始化日历系统时出错: {e}")
            self.events = []
    
    async def save(self):
        """保存日历事件到文件"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
            print("日历事件已保存")
        except Exception as e:
            print(f"保存日历事件时出错: {e}")
    
    async def add_event(self, title, date_time, description="", reminder=True, category="general"):
        """添加新事件到日历
        
        Args:
            title (str): 事件标题
            date_time (str): 事件日期时间，格式为 "YYYY-MM-DD HH:MM"
            description (str): 事件描述
            reminder (bool): 是否需要提醒
            category (str): 事件类别
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 验证日期格式
            event_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M")
            
            # 创建事件
            event = {
                "id": str(uuid.uuid4())[:8],  # 生成简短的唯一ID
                "title": title,
                "date_time": date_time,
                "description": description,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reminder": reminder,
                "category": category,
                "reminded": False  # 记录是否已提醒过
            }
            
            # 添加事件
            self.events.append(event)
            
            # 保存事件
            await self.save()
            
            return True
        except ValueError:
            print(f"日期格式无效: {date_time}")
            return False
        except Exception as e:
            print(f"添加事件时出错: {e}")
            return False
    
    async def delete_event(self, event_id):
        """从日历中删除事件
        
        Args:
            event_id (str): 要删除的事件ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            initial_count = len(self.events)
            self.events = [e for e in self.events if e.get("id") != event_id]
            
            if len(self.events) < initial_count:
                await self.save()
                return True
            else:
                print(f"未找到ID为 {event_id} 的事件")
                return False
        except Exception as e:
            print(f"删除事件时出错: {e}")
            return False
    
    async def get_upcoming_events(self, days=7, category=None):
        """获取即将到来的事件
        
        Args:
            days (int): 要查看的未来天数
            category (str, optional): 如果指定，只返回特定类别的事件
            
        Returns:
            list: 即将到来的事件列表
        """
        try:
            now = datetime.now()
            end_date = now + timedelta(days=days)
            
            upcoming = []
            
            for event in self.events:
                try:
                    event_time = datetime.strptime(event["date_time"], "%Y-%m-%d %H:%M")
                    
                    # 检查事件是否在指定时间范围内
                    if now <= event_time <= end_date:
                        # 检查类别（如果指定）
                        if category is None or event["category"] == category:
                            # 添加人性化的时间描述
                            event = event.copy()  # 创建副本避免修改原事件
                            event["time_desc"] = self.get_human_time_description(event_time)
                            upcoming.append(event)
                except (ValueError, KeyError):
                    # 跳过格式无效的事件
                    continue
            
            # 按时间排序
            upcoming.sort(key=lambda x: datetime.strptime(x["date_time"], "%Y-%m-%d %H:%M"))
            
            return upcoming
        except Exception as e:
            print(f"获取即将到来的事件时出错: {e}")
            return []
    
    async def get_today_events(self):
        """获取今天的事件
        
        Returns:
            list: 今天的事件列表
        """
        try:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            
            today_events = []
            
            for event in self.events:
                try:
                    # 检查事件是否在今天
                    if event["date_time"].startswith(today):
                        event_time = datetime.strptime(event["date_time"], "%Y-%m-%d %H:%M")
                        
                        # 添加人性化的时间描述
                        event = event.copy()
                        event["time_desc"] = self.get_human_time_description(event_time)
                        today_events.append(event)
                except (ValueError, KeyError):
                    # 跳过格式无效的事件
                    continue
            
            # 按时间排序
            today_events.sort(key=lambda x: datetime.strptime(x["date_time"], "%Y-%m-%d %H:%M"))
            
            return today_events
        except Exception as e:
            print(f"获取今天事件时出错: {e}")
            return []
    
    async def clean_past_events(self, days_to_keep=30):
        """清理过去的事件
        
        Args:
            days_to_keep (int): 保留过去多少天的事件
            
        Returns:
            int: 清理的事件数量
        """
        try:
            now = datetime.now()
            cutoff_date = now - timedelta(days=days_to_keep)
            
            initial_count = len(self.events)
            
            # 过滤事件
            self.events = [
                e for e in self.events 
                if datetime.strptime(e["date_time"], "%Y-%m-%d %H:%M") >= cutoff_date
            ]
            
            # 计算删除的事件数量
            removed_count = initial_count - len(self.events)
            
            if removed_count > 0:
                await self.save()
                print(f"已清理 {removed_count} 个过期事件")
                
            return removed_count
        except Exception as e:
            print(f"清理过期事件时出错: {e}")
            return 0
    
    async def check_reminders(self):
        """检查即将到来的事件并触发提醒
        
        Returns:
            list: 需要提醒的事件列表
        """
        if not self.reminders_active:
            return []
            
        try:
            now = datetime.now()
            reminder_horizon = now + timedelta(hours=1)  # 提前1小时提醒
            
            to_remind = []
            
            for event in self.events:
                try:
                    # 跳过已经提醒过的事件
                    if event.get("reminded", False):
                        continue
                        
                    # 检查是否需要提醒
                    if not event.get("reminder", True):
                        continue
                    
                    event_time = datetime.strptime(event["date_time"], "%Y-%m-%d %H:%M")
                    
                    # 检查事件是否在提醒时间范围内
                    if now < event_time <= reminder_horizon:
                        # 标记为已提醒
                        event["reminded"] = True
                        to_remind.append(event)
                except (ValueError, KeyError):
                    # 跳过格式无效的事件
                    continue
            
            # 如果有需要提醒的事件，保存更新
            if to_remind:
                await self.save()
            
            return to_remind
        except Exception as e:
            print(f"检查提醒时出错: {e}")
            return []
    
    def get_human_time_description(self, event_time):
        """获取人性化的时间描述
        
        Args:
            event_time (datetime): 事件时间
            
        Returns:
            str: 人性化的时间描述
        """
        now = datetime.now()
        delta = event_time - now
        
        # 今天内
        if event_time.date() == now.date():
            if delta.total_seconds() < 0:
                return f"今天 {event_time.strftime('%H:%M')}（已过）"
            elif delta.total_seconds() < 3600:  # 1小时内
                minutes = int(delta.total_seconds() / 60)
                return f"今天 {event_time.strftime('%H:%M')}（{minutes}分钟后）"
            else:
                hours = int(delta.total_seconds() / 3600)
                return f"今天 {event_time.strftime('%H:%M')}（{hours}小时后）"
        
        # 明天
        elif event_time.date() == (now.date() + timedelta(days=1)):
            return f"明天 {event_time.strftime('%H:%M')}"
        
        # 后天
        elif event_time.date() == (now.date() + timedelta(days=2)):
            return f"后天 {event_time.strftime('%H:%M')}"
        
        # 一周内
        elif delta.days < 7:
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            weekday = weekday_names[event_time.weekday()]
            return f"{weekday} {event_time.strftime('%H:%M')}"
        
        # 更远的未来
        else:
            return event_time.strftime("%Y-%m-%d %H:%M")
    
    async def toggle_reminders(self, active=None):
        """开启或关闭提醒功能
        
        Args:
            active (bool, optional): 是否激活提醒。如果为None，则切换当前状态
            
        Returns:
            bool: 当前提醒状态
        """
        if active is not None:
            self.reminders_active = active
        else:
            self.reminders_active = not self.reminders_active
            
        return self.reminders_active

async def check_sleep_state():
    """检查是否应该进入休眠状态"""
    global sleep_mode, last_activity_time
    
    while not exit_flag:
        try:
            # 如果当前不在休眠状态，检查是否应该休眠
            if not sleep_mode:
                current_time = time.time()
                idle_time = current_time - last_activity_time
                
                if idle_time > SLEEP_TIMEOUT:
                    print(f"已闲置 {idle_time:.1f} 秒，AI进入休眠状态")
                    sleep_mode = True
                    
                    # 可选：生成一个休眠提示
                    sleep_message = "没有人需要我了，我先休息一会儿。"
                    print(f"AI: {sleep_message}")
                    generate_speech(sleep_message)
            
            # 每秒检查一次
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"休眠状态检查出错: {e}")
            await asyncio.sleep(5)  # 出错后等待5秒再继续

async def check_calendar_reminders(calendar_system):
    """检查日历提醒并通知用户"""
    global exit_flag, sleep_mode
    
    print("日历提醒检查任务已启动")
    
    while not exit_flag:
        try:
            # 每60秒检查一次提醒
            for _ in range(60):
                if exit_flag:
                    break
                await asyncio.sleep(1)
            
            if exit_flag:
                break
            
            # 检查需要提醒的事件
            to_remind = await calendar_system.check_reminders()
            
            if to_remind and not sleep_mode:  # 只在非休眠状态下提醒
                for event in to_remind:
                    reminder_message = f"提醒：您有一个即将开始的事件「{event['title']}」，时间是{event['date_time']}。"
                    print(f"\nAI: {reminder_message}")
                    generate_speech_with_emotion(reminder_message, emotion="excited")
                    
                    # 添加到短期记忆
                    reminder_exchange = {"bot": reminder_message}
                    asyncio.create_task(async_save_short_term_memory(reminder_exchange))
                    
                    # 每个提醒之间暂停一下，避免语音重叠
                    await asyncio.sleep(5)
                    
        except Exception as e:
            print(f"检查日历提醒时出错: {e}")
            traceback.print_exc()
            await asyncio.sleep(10)  # 出错后等待10秒再继续

async def main_async():
    """Main async function"""
    global transcriber, exit_flag
    
    try:
        # 创建任务列表
        tasks = []
        threads = []
        
        # 初始化系统组件
        print("初始化系统组件...")
        
        # 初始化语音识别
        if os.path.exists(REFERENCE_WAV):
            print(f"找到参考音频: {REFERENCE_WAV}")
        else:
            print(f"警告: 未找到参考音频 {REFERENCE_WAV}")
        
        # 启动TTS服务
        tts_thread = threading.Thread(target=start_tts_service)
        tts_thread.daemon = True  # 设为守护线程
        tts_thread.start()
        threads.append(tts_thread)
        print("TTS服务已启动")
        
        # 初始化Whisper
        transcriber = initialize_whisper()
        print("语音识别已初始化")
        
        # 启动音频处理线程
        audio_thread = threading.Thread(target=process_audio_queue)
        audio_thread.daemon = True
        audio_thread.start()
        threads.append(audio_thread)
        print("音频处理线程已启动")
        
        # 启动随机对话线程
        random_chat_thread_handle = threading.Thread(target=random_chat_thread)
        random_chat_thread_handle.daemon = True
        random_chat_thread_handle.start()
        threads.append(random_chat_thread_handle)
        print("随机对话线程已启动")
        
        # 启动主动学习线程
        proactive_learning_thread = threading.Thread(target=proactive_learning_thread_worker)
        proactive_learning_thread.daemon = True
        proactive_learning_thread.start()
        threads.append(proactive_learning_thread)
        print("主动学习线程已启动")
        
        # 加载长期记忆
        long_term_memory = load_chat_history()
        print(f"长期记忆加载完成 ({len(long_term_memory)} 条记录)")
        
        # 加载短期记忆
        short_term_memory = await load_short_term_memory()
        print(f"短期记忆加载完成 ({len(short_term_memory)} 条记录)")
        
        # 初始化记忆管理器
        memory_management_task = asyncio.create_task(manage_memory_size())
        tasks.append(memory_management_task)
        
        # 初始化RAG系统
        print("正在加载知识库...")
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
            print(f"创建知识目录: {KNOWLEDGE_DIR}")
        
        rag_system = RAGSystem(llm, KNOWLEDGE_DIR, CHROMA_DB_DIR)
        await rag_system.initialize()
        print("知识库加载完成")
        
        # 初始化情感智能系统
        print("初始化情感智能系统...")
        emotion_system = EmotionalIntelligenceSystem()
        await emotion_system.initialize()
        print("情感智能系统初始化完成")
        
        # 初始化多模态系统
        print("初始化多模态系统...")
        multimodal_system = MultimodalSystem()
        await multimodal_system.initialize()
        print("多模态系统初始化完成")
        
        # 初始化环境感知系统
        print("初始化环境感知系统...")
        context_system = ContextualAwarenessSystem()
        context_update_task = asyncio.create_task(context_system.update_loop())
        tasks.append(context_update_task)
        print("环境感知系统初始化完成")
        
        # 初始化用户偏好系统
        print("初始化用户偏好系统...")
        preference_tracker = PreferenceTracker()
        await preference_tracker.load()
        print("用户偏好系统初始化完成")
        
        # 加载AI心情状态
        print("加载AI心情状态...")
        await load_mood_state()
        mood_evolution_task = asyncio.create_task(mood_evolution_loop())
        tasks.append(mood_evolution_task)
        print("AI心情状态加载完成")
        
        # 加载语音偏好
        print("加载语音偏好...")
        await load_voice_preferences()
        print("语音偏好加载完成")
        
        # 初始化日历系统
        print("初始化日历系统...")
        calendar_system = CalendarSystem()
        await calendar_system.initialize()
        print("日历系统初始化完成")
        
        # 启动每日总结调度器
        daily_summary_task = asyncio.create_task(daily_summary_scheduler())
        tasks.append(daily_summary_task)
        print("每日总结调度器已启动")
        
        # 启动休眠状态检查任务
        sleep_check_task = asyncio.create_task(check_sleep_state())
        tasks.append(sleep_check_task)
        print("休眠状态检查任务已启动")
        
        # 启动日历提醒检查任务
        calendar_reminder_task = asyncio.create_task(check_calendar_reminders(calendar_system))
        tasks.append(calendar_reminder_task)
        print("日历提醒检查任务已启动")
        
        # 创建命令解析器
        command_parser = CommandParser()
        
        # 设置信号处理 - Windows兼容方式
        if os.name == 'posix':  # For Unix/Linux/Mac
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(tasks, threads)))
        else:  # For Windows
            # Windows doesn't support asyncio signal handlers
            # We'll rely on KeyboardInterrupt exception handling in the main() function
            pass
        
        # 显示欢迎信息
        print("\n====================================")
        print("       01 AI助手系统已启动")
        print("====================================")
        print("输入 'help' 查看可用命令")
        print("输入 'quit' 退出程序")
        print("------------------------------------")
        
        # 处理正在休眠状态的提示
        if sleep_mode:
            print("AI当前处于休眠状态，请说或输入'早安01'唤醒")
        
        # 主循环
        while not exit_flag:
            try:
                # 显示提示并等待输入
                command = await async_input("Type here > ")
                
                # 检查是否为空输入
                if not command.strip():
                    continue
                
                # 更新最后活动时间
                last_activity_time = time.time()
                
                # 检查输入是否是命令
                if command_parser.is_command(command):
                    await process_command(
                        command, 
                        command_parser, 
                        tasks, 
                        context_system, 
                        preference_tracker
                    )
                else:
                    # 处理对话
                    user_exchange = {"user": command}
                    await async_save_short_term_memory(user_exchange)
                    
                    await process_chat_input(
                        command, 
                        emotion_system, 
                        multimodal_system, 
                        context_system, 
                        preference_tracker
                    )
            
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                print(f"处理输入时出错: {e}")
                traceback.print_exc()
                continue
        
        # 清理并保存数据
        print("正在保存数据...")
        await consolidate_memories()
        await preference_tracker.save()
        await emotion_system.save()
        await save_mood_state()
        await save_voice_preferences()
        await calendar_system.save()
        
        print("程序结束。")
        
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
        
    finally:
        # 确保所有任务都被取消
        for task in tasks:
            if not task.done():
                task.cancel()

async def extract_calendar_events_from_text(text, calendar_system):
    """从文本中提取可能的日历事件并添加到日历
    
    Args:
        text (str): 用户输入的文本
        calendar_system (CalendarSystem): 日历系统实例
        
    Returns:
        bool: 是否成功添加事件
    """
    try:
        # 构建提示
        prompt = f"""
        分析以下文本，如果包含日程安排或需要添加到日历的事件，请按以下JSON格式返回:
        
        {{
          "is_event": true,  // 如果文本中包含日历事件则为true，否则为false
          "title": "事件标题",
          "date_time": "YYYY-MM-DD HH:MM",  // 事件时间，格式为 YYYY-MM-DD HH:MM
          "description": "事件描述"
        }}
        
        如果不确定具体时间，但有日期，请使用默认时间如上午9点。
        如果没有指定年份，使用当前年份。
        如果没有指定月份和日期，但提到"今天"、"明天"、"后天"等，根据当前日期计算。
        如果文本中没有明确的日历事件，返回 {{"is_event": false}}
        
        文本: "{text}"
        当前日期: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        # 解析响应
        try:
            result = json.loads(response.choices[0].message.content)
            
            if result.get("is_event", False):
                title = result.get("title", "")
                date_time = result.get("date_time", "")
                description = result.get("description", "")
                
                if title and date_time:
                    # 添加到日历
                    success = await calendar_system.add_event(
                        title=title,
                        date_time=date_time,
                        description=description
                    )
                    
                    if success:
                        return {
                            "added": True,
                            "title": title,
                            "date_time": date_time
                        }
            
            return {"added": False}
            
        except (json.JSONDecodeError, KeyError):
            print("解析AI响应失败")
            return {"added": False}
            
    except Exception as e:
        print(f"从文本提取日历事件时出错: {e}")
        traceback.print_exc()
        return {"added": False}
    
def main():
    """
    Main entry point for the 01 AI Companion
    
    Features implemented:
    - Voice interaction with ASR and TTS
    - LLM-based conversation with memory
    - Emotional intelligence system for emotion detection and response
    - Multimodal support for image processing
    - Contextual awareness of time, date, and environment
    - User preference tracking and personalization
    - Proactive learning system
    - Daily summaries and activity suggestions
    - Dynamic AI mood states
    """
    # 确保导入json模块
    import json
    
    try:
        # Run the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断。正在关闭...")
        global exit_flag
        exit_flag = True
        # 整合记忆并保存
        try:
            # Create and run a new event loop for consolidation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(consolidate_memories())
            print("对话历史已整合并保存!")
            loop.close()
        except Exception as e:
            print(f"保存对话历史时出错: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        exit_flag = True
    finally:
        # Give threads time to clean up
        time.sleep(1)

if __name__ == '__main__':
    main()