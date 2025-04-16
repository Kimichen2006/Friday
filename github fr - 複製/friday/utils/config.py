import os
import yaml

def load_config(config_path=None):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str, optional): Path to config file. If None, tries to use environment variable.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Check if config path is in environment variable
        config_path = os.environ.get('AI_GIRLFRIEND_CONFIG', 'config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        # Return default configuration
        return {
            'llm': {
                'model': 'default_model',
                'api_key': None
            },
            'memory': {
                'long_term_file': 'long_term_memory.json',
                'short_term_file': 'short_term_memory.json',
                'max_size': 20000
            },
            'audio': {
                'sample_rate': 16000,
                'silence_threshold': 0.03,
                'silence_duration': 0.5
            },
            'tts': {
                'service_path': None,
                'reference_wav': None,
                'gpt_model': None,
                'sovits_model': None
            },
            'rag': {
                'knowledge_dir': 'knowledge_base',
                'db_dir': 'chroma_db'
            }
        }
