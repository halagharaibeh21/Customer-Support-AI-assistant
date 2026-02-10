"""
Script to Create/Rebuild lazaboon_chroma_db with correct embeddings
This will create a database compatible with your chatbot
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Configuration
INPUT_FILE = "companyinfo.txt"  # Your source file
OUTPUT_DIR = "./lazaboon_chroma_db"  # Output directory
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Step 1: Check if input file exists
print(f"\n[STEP 1] Checking for input file: {INPUT_FILE}")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found!")
    print("Please make sure companyinfo.txt is in the same folder as this script.")
    exit(1)
print(f"Found {INPUT_FILE}")

# Step 2: Read the file
print(f"\n[STEP 2] Reading {INPUT_FILE}...")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        document_text = f.read()
    print(f"File loaded successfully ({len(document_text)} characters)")
except Exception as e:
    print(f"ERROR reading file: {e}")
    exit(1)

# Step 3: Split into chunks
print(f"\n[STEP 3] Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)
chunks = text_splitter.split_text(document_text)
print(f"Created {len(chunks)} chunks")

# Step 4: Load embedding model
print(f"\n[STEP 4] Loading embedding model...")
print("This may take a few minutes on first run (downloading model)...")
try:
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )
    print("Embedding model loaded successfully")
except Exception as e:
    print(f"ERROR loading embedding model: {e}")
    exit(1)

# Step 5: Delete old database if it exists
print(f"\n[STEP 5] Preparing output directory...")
if os.path.exists(OUTPUT_DIR):
    print(f"{OUTPUT_DIR} already exists. Deleting old database...")
    import shutil
    try:
        shutil.rmtree(OUTPUT_DIR)
        print("Old database deleted")
    except Exception as e:
        print(f"ERROR deleting old database: {e}")
        print("Please manually delete the lazaboon_chroma_db folder and try again.")
        exit(1)

# Step 6: Create new database
print(f"\n[STEP 6] Creating new Chroma database...")
print("This may take a few minutes (embedding all chunks)...")
try:
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=OUTPUT_DIR,
        metadatas=[{"source": "companyinfo.txt", "chunk": i} for i in range(len(chunks))]
    )
    print("✓ Database created successfully!")
except Exception as e:
    print(f"ERROR creating database: {e}")
    exit(1)

# Step 7: Verify the database
print(f"\n[STEP 7] Verifying database...")
try:
    # Reload to verify
    vectorstore_check = Chroma(
        persist_directory=OUTPUT_DIR,
        embedding_function=embedding
    )
    
    # Get document count
    collection = vectorstore_check._collection
    doc_count = collection.count()
    
    print(f"Database verification successful!")
    print(f"  → Total documents in database: {doc_count}")
    
    if doc_count == 0:
        print("WARNING: Database was created but contains 0 documents!")
        print("This should not happen. Please check the input file.")
    else:
        # Test a simple query
        print(f"\n[STEP 8] Testing retrieval...")
        test_results = vectorstore_check.similarity_search("company", k=1)
        if len(test_results) > 0:
            print(f"Test query successful!")
            print(f"  Sample result preview: {test_results[0].page_content[:100]}...")
        else:
            print("Test query returned no results")
            
except Exception as e:
    print(f"ERROR verifying database: {e}")
    exit(1)

print("\n" + "=" * 80)
print("DATABASE CREATION COMPLETE!")
print("=" * 80)
print(f"""
Summary:
- Input file: {INPUT_FILE}
- Output directory: {OUTPUT_DIR}
- Total chunks: {len(chunks)}
- Documents in database: {doc_count}
- Embedding model: intfloat/multilingual-e5-base

Next steps:
1. Your chatbot should now be able to find documents
2. Run your chatbot and test with queries
3. If still having issues, run the diagnostic script again
""")
