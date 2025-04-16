# Using Local Models with Ollama in Friday

This guide explains how to use Ollama to run local LLM models with Friday instead of using OpenRouter API.

## What is Ollama?

[Ollama](https://ollama.ai/) lets you run large language models locally on your computer. This gives you:
- Complete privacy - your data never leaves your device
- No API costs or rate limits
- Offline functionality
- Control over which models and versions you use

## Setup Instructions

### 1. Install Ollama

First, download and install Ollama from [https://ollama.ai/](https://ollama.ai/).

### 2. Pull a Model

After installing Ollama, pull a model you want to use. For example:

```bash
# Open a terminal/command prompt
ollama pull llama2
```

Some recommended models for Friday:
- `llama2` - A good all-around model
- `mistral` - Good performance with less resources
- `neural-chat` - Optimized for conversation
- `llama2-uncensored` - For more unrestricted responses

### 3. Create a New Adapter for Ollama in Friday

Create a file `friday/models/ollama_adapter.py` with the following content:

```python
import requests
import json

class OllamaLLM:
    """Adapter for using Ollama local LLM models"""
    def __init__(self, model_name="llama2", host="http://localhost:11434"):
        self.model = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        
    async def ainvoke(self, prompt):
        """Invoke Ollama API asynchronously"""
        try:
            # For simplicity, we're using a synchronous request in an async function
            # In production, use aiohttp for truly async operation
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data.get("response", "")
            return OllamaResponse(content)
        except Exception as e:
            print(f"Ollama API error: {e}")
            return OllamaResponse("Sorry, I'm having trouble responding right now.")

class OllamaResponse:
    """Simple response wrapper to match the expected interface"""
    def __init__(self, content):
        self.content = content
```

### 4. Modify the LLM Initialization Code

Now update `friday/models/llm.py` to include the Ollama adapter:

```python
from openai import AsyncOpenAI
from friday.models.ollama_adapter import OllamaLLM  # Import the new adapter

def initialize_llm(config):
    """
    Initialize the LLM based on configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        object: Initialized LLM instance
    """
    # Check if we should use Ollama
    use_ollama = config['llm'].get('use_ollama', False)
    
    if use_ollama:
        model_name = config['llm'].get('ollama_model', 'llama2')
        host = config['llm'].get('ollama_host', 'http://localhost:11434')
        print(f"Using local Ollama model: {model_name}")
        return OllamaLLM(model_name, host)
    else:
        # Original OpenRouter code
        model = config['llm'].get('model', 'qwen/qwen2.5-7b-instruct:free')
        api_key = config['llm'].get('api_key', '')
        
        if not api_key:
            print("Warning: No API key provided for LLM. Using a mock LLM instead.")
            return MockLLM()
        
        return OpenRouterLLM(api_key, model)
```

### 5. Update Your Configuration File

Add Ollama settings to your `config.yaml`:

```yaml
# LLM (Large Language Model) settings
llm:
  # OpenRouter settings (will be ignored if use_ollama is true)
  model: "qwen/qwen2.5-7b-instruct:free"
  api_key: ""
  
  # Ollama settings
  use_ollama: true
  ollama_model: "llama2"  # Choose from models you've pulled
  ollama_host: "http://localhost:11434"
```

## Example Usage Scenarios

### Basic Usage

1. Start the Ollama service (it should run automatically after installation)
2. Set `use_ollama: true` in your config.yaml
3. Run Friday normally: `python main.py`

Friday will now use your local Ollama model for all interactions.

### Switching Between Models

You can easily switch between different local models by:

1. Pulling the model first (if you haven't already):
   ```bash
   ollama pull mistral
   ```

2. Update your config.yaml:
   ```yaml
   llm:
     use_ollama: true
     ollama_model: "mistral"
   ```

3. Restart Friday

### Custom Prompt Templates

Ollama models sometimes work better with specific prompt templates. You can modify `friday/rag/system.py` to use model-specific prompts:

```python
def get_prompt_template(model_name):
    """Get an appropriate prompt template for the model"""
    if "llama" in model_name.lower():
        return """
        <|system|>
        You are Friday, an AI assistant. You are helpful, informative, and accurate.
        Always try to provide the most relevant and useful information to the user.
        If you don't know something, just say so rather than making up information.
        
        Conversation history:
        {chat_history}
        
        Relevant context:
        {context}
        </|system|>
        
        <|user|>
        {question}
        </|user|>
        
        <|assistant|>
        """
    else:
        # Default prompt format for other models
        return """
        You are Friday, an AI assistant. You are helpful, informative, and accurate.
        Always try to provide the most relevant and useful information to the user.
        If you don't know something, just say so rather than making up information.
        
        Conversation history:
        {chat_history}
        
        Relevant context:
        {context}
        
        User: {question}
        
        Friday:
        """
```

## Performance Considerations

Local models may be slower or faster than cloud APIs depending on your hardware:

- **GPU Acceleration**: If you have a supported NVIDIA GPU, Ollama will use it automatically
- **Memory Requirements**: Larger models need more RAM (8GB+ recommended, 16GB+ preferred)
- **Model Size vs. Quality**: Smaller models run faster but may give lower quality responses
- **Quantization**: Models with 'q4' in their name (e.g., llama2:7b-q4) use quantization for better performance

## Troubleshooting

- **Ollama Not Responding**: Make sure the Ollama service is running
- **Out of Memory**: Try a smaller model or a quantized version
- **Poor Responses**: Different models have different strengths; try another model
- **Model Loading Failed**: Check for sufficient disk space and correct model name

Remember that you can always switch back to OpenRouter by setting `use_ollama: false` in your config file. 