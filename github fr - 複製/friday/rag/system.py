import os
import json
import asyncio
import random

class RAGSystem:
    """Retrieval-Augmented Generation system for AI assistant"""
    def __init__(self, llm, config):
        """
        Initialize RAG system
        
        Args:
            llm: Language model for generating responses
            config (dict): Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.knowledge_dir = config['rag'].get('knowledge_dir', 'knowledge_base')
        self.db_dir = config['rag'].get('db_dir', 'chroma_db')
        self.character_prompt = """
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
        
    async def initialize(self):
        """Initialize the RAG system and knowledge base"""
        print("Initializing RAG system...")
        
        # Create knowledge directory if it doesn't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            print(f"Created knowledge base directory: {self.knowledge_dir}")
            # Create example document
            with open(os.path.join(self.knowledge_dir, "sample.txt"), "w", encoding="utf-8") as f:
                f.write("This is a sample document for the RAG system. Replace with actual knowledge base content.")
        
        print("RAG system initialized!")
    
    async def get_answer(self, query, history=None):
        """
        Get answer using RAG approach
        
        Args:
            query (str): User query
            history (list, optional): Chat history
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare chat history context
            history_text = ""
            
            # Try to get chat history from memory
            try:
                from friday.models.memory import load_chat_history
                chat_history = await load_chat_history(self.config)
                
                # Get recent exchanges for context (up to 10)
                if chat_history:
                    # Sort by timestamp if available
                    sorted_history = sorted(
                        chat_history, 
                        key=lambda x: x.get("timestamp", ""),
                        reverse=True
                    )
                    
                    recent_history = sorted_history[:10]
                    for item in recent_history:
                        if "user" in item:
                            history_text += f"User: {item['user']}\n"
                        if "bot" in item:
                            history_text += f"Friday: {item['bot']}\n"
            except Exception as e:
                print(f"Error loading chat history: {e}")
                # If history loading fails, use empty history
                history_text = ""
            
            # Get knowledge context (simplified)
            context_text = ""
            try:
                # Load random documents as context
                if os.path.exists(self.knowledge_dir):
                    files = [f for f in os.listdir(self.knowledge_dir) if f.endswith('.txt')]
                    if files:
                        sample_files = random.sample(files, min(2, len(files)))
                        for file in sample_files:
                            with open(os.path.join(self.knowledge_dir, file), 'r', encoding='utf-8') as f:
                                context_text += f.read() + "\n\n"
            except Exception as e:
                print(f"Error loading knowledge base content: {e}")
            
            # Create prompt with context and history
            full_prompt = self.character_prompt.format(
                chat_history=history_text,
                context=context_text,
                question=query
            )
            
            # Get LLM response
            response = await self.llm.ainvoke(full_prompt)
            
            # Save to short-term memory
            try:
                # Import here to avoid circular imports
                from friday.models.memory import async_save_short_term_memory
                
                user_exchange = {"user": query}
                bot_exchange = {"bot": response.content}
                
                # Asynchronously save to short-term memory
                asyncio.create_task(async_save_short_term_memory(
                    user_exchange,
                    self.config['memory'].get('short_term_file', 'short_term_memory.json')
                ))
                asyncio.create_task(async_save_short_term_memory(
                    bot_exchange,
                    self.config['memory'].get('short_term_file', 'short_term_memory.json')
                ))
            except Exception as e:
                print(f"Error saving to short-term memory: {e}")
                
            return response.content
            
        except Exception as e:
            print(f"Error getting answer from RAG system: {e}")
            import traceback
            print(traceback.format_exc())
            return f"I encountered an error processing your request: {str(e)}"
