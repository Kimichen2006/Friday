from openai import AsyncOpenAI

class OpenRouterLLM:
    """OpenRouter LLM adapter class for using OpenRouter API"""
    def __init__(self, api_key, model):
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
    async def ainvoke(self, prompt):
        """Invoke OpenRouter API asynchronously"""
        try:
            completion = await self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Friday-AI-Assistant",
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            return OpenRouterResponse(completion.choices[0].message.content)
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return OpenRouterResponse("Sorry, I'm having trouble responding right now.")

class OpenRouterResponse:
    """Simple response wrapper to match the expected interface"""
    def __init__(self, content):
        self.content = content

def initialize_llm(config):
    """
    Initialize the LLM based on configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        object: Initialized LLM instance
    """
    model = config['llm'].get('model', 'qwen/qwen2.5-7b-instruct:free')
    api_key = config['llm'].get('api_key', '')
    
    # Default to OpenRouter LLM if no specific type is specified
    if not api_key:
        print("Warning: No API key provided for LLM. Using a mock LLM instead.")
        return MockLLM()
    
    return OpenRouterLLM(api_key, model)

class MockLLM:
    """Mock LLM for testing without API access"""
    async def ainvoke(self, prompt):
        """Return a mock response"""
        return OpenRouterResponse("This is a mock response as no API key was provided. Please update your configuration with a valid API key.")
