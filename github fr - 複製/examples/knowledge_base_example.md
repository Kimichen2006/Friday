# Creating Custom Knowledge Bases for Friday

This guide shows you how to create, organize, and use custom knowledge bases to make Friday more knowledgeable about specific topics.

## Understanding the RAG System

Friday uses a Retrieval-Augmented Generation (RAG) system, which:
1. Retrieves relevant information from a knowledge base
2. Augments the LLM's prompt with this information
3. Generates more informed responses using this context

## Basic Knowledge Base Setup

### 1. Configure Your Knowledge Directory

First, set the knowledge directory in your `config.yaml`:

```yaml
# RAG (Retrieval-Augmented Generation) settings
rag:
  knowledge_dir: "knowledge_base"  # This is the directory where your knowledge files will be stored
  db_dir: "chroma_db"
```

### 2. Create Knowledge Files

Create a directory structure for your knowledge:

```
knowledge_base/
├── general/
│   ├── about_friday.txt
│   └── common_commands.txt
├── personal/
│   ├── family_members.txt
│   └── preferences.txt
├── technical/
│   ├── programming_tips.txt
│   └── computer_maintenance.txt
└── interests/
    ├── cooking_recipes.txt
    └── movie_recommendations.txt
```

### 3. Write Knowledge Content

Here's an example of what `knowledge_base/personal/family_members.txt` might contain:

```
# Family Members

Jack is my husband. We've been married for 5 years. His birthday is on March 15th.
He works as a software engineer and loves hiking on weekends.

Emma is our daughter. She's 3 years old and attends Little Stars Preschool.
Her favorite color is purple and she loves dinosaurs, especially T-Rex.

Max is our golden retriever. He's 2 years old and loves playing fetch in the park.
```

## Advanced Knowledge Base Techniques

### Formatting for Better Retrieval

Structure your knowledge files to help the RAG system retrieve relevant information:

1. **Use clear headings and sections**:
   ```
   # Cooking Recipes
   
   ## Chocolate Chip Cookies
   Ingredients:
   - 2 1/4 cups all-purpose flour
   - 1 teaspoon baking soda
   - ...
   
   ## Pasta Carbonara
   Ingredients:
   - 8 oz spaghetti
   - 2 large eggs
   - ...
   ```

2. **Include synonyms and alternative phrasing**:
   ```
   # Computer Troubleshooting
   
   ## Internet Connection Issues (Network Problems, WiFi not working)
   If your internet connection isn't working (offline, no connectivity), try these steps:
   ```

3. **Prioritize important information**:
   ```
   IMPORTANT: Mom's medication needs to be taken twice daily, at 9 AM and 9 PM with food.
   ```

### Creating a Custom RAG Implementation

For more sophisticated knowledge retrieval, you can modify `friday/rag/system.py` to implement vector embeddings:

```python
# Add to the imports
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGSystem:
    def __init__(self, llm, config):
        # Existing initialization code...
        
        # Add embedding model for better retrieval
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.using_embeddings = True
            print("Using semantic search for knowledge retrieval")
        except:
            self.using_embeddings = False
            print("Sentence transformers not available, falling back to keyword search")
    
    async def find_relevant_documents(self, query, limit=2):
        """Find the most relevant documents for a query using embeddings"""
        if not self.using_embeddings:
            # Fall back to existing random document selection
            return await self.get_random_documents(limit)
            
        # Encode the query to a vector
        query_embedding = self.embedding_model.encode(query)
        
        relevant_docs = []
        all_files = []
        
        # Get all knowledge files
        for root, _, files in os.walk(self.knowledge_dir):
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))
        
        # Calculate scores for each file
        scores = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks if content is large
                chunks = self.split_into_chunks(content)
                
                # Find the best matching chunk
                chunk_embeddings = self.embedding_model.encode(chunks)
                chunk_scores = np.dot(chunk_embeddings, query_embedding)
                best_score = np.max(chunk_scores)
                
                scores.append((file_path, best_score))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Sort by score and get the top results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_files = scores[:limit]
        
        # Load the content of top files
        for file_path, score in top_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                relevant_docs.append(content)
            except Exception as e:
                print(f"Error reading top file {file_path}: {e}")
        
        return relevant_docs
    
    def split_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Split a long text into overlapping chunks"""
        chunks = []
        if len(text) <= chunk_size:
            return [text]
            
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            
            if i + chunk_size >= len(text):
                break
                
        return chunks
```

## Real-World Examples

### Personal Assistant Knowledge Base

Create `knowledge_base/personal/schedule.txt`:

```
# Weekly Schedule

## Monday
- 8:00 AM: Team meeting
- 12:00 PM: Lunch with Sarah
- 3:00 PM: Dentist appointment

## Tuesday
- 9:00 AM: Call with client
- 5:30 PM: Gym class

## Wednesday
- Work from home day
- 2:00 PM: Virtual conference

## Thursday
- 10:00 AM: Project review
- 7:00 PM: Dinner with parents

## Friday
- 11:00 AM: One-on-one with manager
- 4:00 PM: Happy hour with team
```

### Technical Documentation

Create `knowledge_base/technical/home_network.txt`:

```
# Home Network Configuration

## WiFi Networks
- Main Network: "HomeNetwork5G" (5GHz, faster but shorter range)
  - Password: UseTheForce123
  - Devices: Laptops, phones, streaming devices
  
- Extended Network: "HomeNetwork2G" (2.4GHz, better range)
  - Password: UseTheForce123
  - Devices: IoT devices, smart home gadgets

## IP Addresses
- Router: 192.168.1.1
- NAS Device: 192.168.1.5
- Printer: 192.168.1.10
- Static IP range: 192.168.1.50 - 192.168.1.100

## Troubleshooting
When internet is down:
1. Restart router by unplugging for 30 seconds
2. Check if both lights are solid blue
3. If problem persists, call ISP at 555-123-4567
```

### Cooking Recipes Knowledge Base

Create `knowledge_base/interests/quick_meals.txt`:

```
# Quick Meal Recipes

## 15-Minute Pasta
Ingredients:
- 8 oz pasta
- 2 tbsp olive oil
- 2 cloves garlic, minced
- 1 can diced tomatoes
- Salt and pepper to taste
- Fresh basil

Instructions:
1. Boil pasta according to package instructions
2. In a pan, heat olive oil and sauté garlic until fragrant
3. Add tomatoes, simmer for 5 minutes
4. Drain pasta, mix with sauce
5. Season and top with fresh basil

## Microwave Baked Potato
Ingredients:
- 1 large russet potato
- 1 tbsp butter
- Salt and pepper
- Optional toppings: cheese, sour cream, chives

Instructions:
1. Wash potato and poke several holes with a fork
2. Microwave on high for 5 minutes, turn over, microwave 5 more minutes
3. Cut open and fluff with fork
4. Add butter, salt, pepper, and desired toppings
```

## Best Practices

1. **Regularly update your knowledge base** as information changes
2. **Structure information hierarchically** with clear sections and subsections
3. **Include specific details** rather than vague statements
4. **Run `rag rebuild`** command after making changes to your knowledge base
5. **Test with questions** to ensure Friday can retrieve the information correctly

By following this guide, you can create a rich, personalized knowledge base that allows Friday to give you highly specific and relevant information about your life, interests, and work. 