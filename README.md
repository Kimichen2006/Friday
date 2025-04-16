# Friday
An intergeration of different models with new RAG memory system
# Friday - Voice-Interactive AI Assistant

Friday is a sophisticated, voice-interactive AI companion that uses Large Language Models (LLMs) and Text-to-Speech technology to create a natural conversational experience.

## Features

- **Voice Recognition**: Interact with Friday through speech using whisper for accurate transcription
- **Text-to-Speech**: Natural voice responses using GPT-SoVITS
- **Memory Management**: Stores conversation history with importance-based priority system
- **RAG System**: Retrieval-Augmented Generation for knowledge-based responses
- **Random Chat**: Friday can initiate conversations on its own at random intervals
- **Command Interface**: Easy-to-use text commands for controlling various features

## Requirements

- Python 3.8 or higher
- OpenRouter API key (for LLM access)
- GPT-SoVITS for text-to-speech (optional)
- At least 4GB RAM for running whisper models
- Windows, macOS, or Linux

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd friday
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the system by editing `config.yaml`:
   ```yaml
   # LLM (Large Language Model) settings
   llm:
     model: "qwen/qwen2.5-7b-instruct:free"  # Model name on OpenRouter
     api_key: "your-openrouter-api-key"      # Your OpenRouter API key

   # Memory management settings
   memory:
     long_term_file: "long_term_memory.json"
     short_term_file: "short_term_memory.json"
     max_size: 20000  # Maximum number of entries to keep

   # Text-to-Speech settings (optional)
   tts:
     service_path: "path/to/GPT-SoVITS-directory"
     reference_wav: "path/to/reference-audio.wav"
     gpt_model: "path/to/GPT-SoVITS/model.ckpt"
     sovits_model: "path/to/SoVITS/model.pth"
   ```

## Usage

Run the assistant:
```
python main.py
```

### Available Commands

- `voice on` - Enable voice recognition mode
- `voice off` - Disable voice recognition mode
- `random chat on` - Enable random conversation initiation
- `random chat off` - Disable random conversation initiation
- `rag rebuild` - Rebuild RAG knowledge base index
- `save` - Manually save current conversation history
- `memory trim` - Manually optimize memory capacity
- `quit` - Exit program
- `help` - Show help message

Any other text input will be sent to the AI for processing.

## Project Structure

- `main.py` - Main application entry point
- `config.yaml` - Configuration file
- `friday/` - Core modules
  - `audio/` - Audio processing modules
    - `recorder.py` - Records voice input
    - `processor.py` - Processes audio recordings
    - `tts.py` - Text-to-speech functionality
  - `models/` - AI models
    - `llm.py` - LLM integration
    - `memory.py` - Memory management
  - `rag/` - Retrieval Augmented Generation
    - `system.py` - Knowledge retrieval system
  - `utils/` - Utility functions
    - `config.py` - Configuration loading
  - `web/` - Web-related functionality (optional)

## Knowledge Base

Friday uses a Retrieval-Augmented Generation (RAG) system to access knowledge. You can add your own knowledge by creating text files in the `knowledge_base/` directory.

## Memory System

Friday has both short-term and long-term memory:
- Short-term memory stores recent conversations
- Long-term memory stores important information with a priority system
- Memories are automatically consolidated and trimmed to maintain performance

## Voice Recognition

Friday uses the whisper model for speech recognition. For better performance:
- Speak clearly with minimal background noise
- Use a good quality microphone
- Allow a short pause after speaking

## Text-to-Speech (Optional)

Friday supports GPT-SoVITS for high-quality text-to-speech. To use it:
1. Install GPT-SoVITS separately
2. Configure the paths in `config.yaml`
3. Provide a reference audio file

## Advanced Configuration

For better performance with whisper, the system automatically sets:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

This prevents OpenMP runtime conflicts when using whisper.

## Troubleshooting

### Common Issues

1. **Voice recognition not working**
   - Check if you've enabled it with `voice on`
   - Ensure whisper is properly installed
   - Check microphone permissions and setup

2. **Text-to-speech not working**
   - Verify GPT-SoVITS paths in the configuration
   - Ensure the GPT-SoVITS service is running

3. **Memory errors**
   - Try `memory trim` to reduce memory usage
   - Check if memory files are corrupted

4. **LLM not responding**
   - Verify your OpenRouter API key
   - Check internet connection
   - Ensure the model name is correct

## License

MIT License

## Acknowledgments

- OpenAI for whisper
- OpenRouter for LLM API access
- GPT-SoVITS for text-to-speech capability
- browser-use for browser
