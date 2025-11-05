import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- 1. Load Document ---
print("Loading document...")
loader = TextLoader("data.txt")
documents = loader.load()

# --- 2. Split Document ---
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

if not texts:
    print("No text found in 'data.txt'. Exiting.")
    exit()

# --- 3. Create Embeddings & Vector Store ---
print("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# Set a reasonable batch size (Google's API limit is 100)
batch_size = 90
db = None

try:
    # Create the vector store with the first batch
    first_batch = texts[0:batch_size]
    print(f"Creating vector store with initial batch (0 to {len(first_batch)})...")
    db = FAISS.from_documents(first_batch, embeddings)

    # Now, add the rest of the batches
    for i in range(batch_size, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Adding batch {i} to {i+len(batch)}...")
        
        # Use add_documents to add to the existing index
        db.add_documents(batch) 
        
        # Optional: Add a small delay if you still see rate limit errors
        # time.sleep(1) 

    # --- 4. Save the final vector store ---
    db.save_local("faiss_index")
    print("\nDone. Vector store saved as 'faiss_index'")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"{e}")
    print("\nThis was likely a network timeout or API issue.")
    print("Please check your firewall/VPN settings and try running the script again.")