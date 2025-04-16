import json
import os
import asyncio
from datetime import datetime, timezone

async def load_chat_history(config):
    """
    Load chat history from file
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: Loaded chat history
    """
    filename = config['memory']['long_term_file']
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error reading chat history: {e}")
        return []

async def async_save_short_term_memory(exchange, filename='short_term_memory.json'):
    """
    Save a single exchange to short-term memory asynchronously
    
    Args:
        exchange (dict): The exchange to add to short-term memory
        filename (str): Path to save the short-term memory file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
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
                
            print(f"Saved exchange to short-term memory: {filename}")
    
    except Exception as e:
        print(f"Error saving to short-term memory: {e}")
        import traceback
        print(traceback.format_exc())

async def consolidate_memories(config):
    """
    Evaluate short-term memory priorities and consolidate with long-term memory
    """
    try:
        print("Consolidating memories...")
        
        # Load both memory stores
        short_term_file = config['memory'].get('short_term_file', 'short_term_memory.json')
        long_term_file = config['memory'].get('long_term_file', 'long_term_memory.json')
        
        short_term = []
        if os.path.exists(short_term_file):
            with open(short_term_file, 'r', encoding='utf-8') as f:
                short_term = json.load(f)
        
        long_term = []
        if os.path.exists(long_term_file):
            with open(long_term_file, 'r', encoding='utf-8') as f:
                long_term = json.load(f)
        
        if not short_term:
            print("No short-term memories to consolidate")
            return
        
        # Reconsider priorities for all short-term memories
        for item in short_term:
            if "bot" in item:
                # Simple priority determination (could be replaced with more sophisticated logic)
                priority = await determine_importance(item["bot"])
                item["priority"] = priority
        
        # Combine memories
        combined_memory = long_term + short_term
        
        # Save combined memory to long-term file
        with open(long_term_file, 'w', encoding='utf-8') as f:
            json.dump(combined_memory, f, ensure_ascii=False, indent=2)
        
        # Clear short-term memory
        with open(short_term_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        print(f"Memory consolidation complete! Transferred {len(short_term)} memories to long-term storage.")
    
    except Exception as e:
        print(f"Error consolidating memories: {e}")

async def determine_importance(text):
    """
    Analyze text content to determine its importance level
    
    Args:
        text (str): The text to analyze
    
    Returns:
        int: Importance level (1=low, 2=medium, 3=high, 4=permanent)
    """
    # Simple importance determination
    if len(text) > 200:
        return 3  # Longer responses are more important
    elif len(text) > 100:
        return 2  # Medium length responses
    else:
        return 1  # Short responses are less important

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
    
    print(f"Memory exceeds threshold ({len(memory_list)}>{max_size}), optimizing...")
    
    # Create a copy to avoid modifying during iteration
    working_memory = memory_list.copy()
    
    # Sort by priority (ascending) and timestamp (ascending)
    # This means lower priority and older items come first (to be deleted)
    working_memory.sort(key=lambda x: (x.get("priority", 1), x.get("timestamp", "")))
    
    # Keep only the most recent/important entries
    trimmed = working_memory[len(working_memory) - max_size:]
    
    print(f"Memory optimization complete: {len(memory_list)} -> {len(trimmed)}")
    return trimmed

async def manage_memory_size(config):
    """Periodically check memory size and trim if needed"""
    print("Memory management thread started")
    check_interval = 600  # Check every 10 minutes
    
    while True:
        try:
            # Load current memory
            long_term_file = config['memory'].get('long_term_file', 'long_term_memory.json')
            max_size = config['memory'].get('max_size', 20000)
            
            if os.path.exists(long_term_file):
                with open(long_term_file, 'r', encoding='utf-8') as f:
                    long_term = json.load(f)
                
                # Only trim if we have significant memory
                if len(long_term) > max_size:
                    print(f"Current memory size: {len(long_term)}, performing optimization...")
                    
                    # Trim long-term memory
                    trimmed_long_term = await trim_memory(long_term, max_size)
                    
                    # Save trimmed memory
                    with open(long_term_file, 'w', encoding='utf-8') as f:
                        json.dump(trimmed_long_term, f, ensure_ascii=False, indent=2)
                    
                    print(f"Long-term memory optimization complete: {len(long_term)} -> {len(trimmed_long_term)}")
            
            # Wait for next check
            await asyncio.sleep(check_interval)
                
        except Exception as e:
            print(f"Error during memory management: {e}")
            await asyncio.sleep(60)  # Wait before retrying
