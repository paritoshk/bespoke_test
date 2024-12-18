# scripts/download_ccnews.py

from datasets import load_dataset
import gzip
import random
from tqdm import tqdm

def download_and_prepare_documents(year="2024", num_docs=40000):
    """
    Download documents from CC News dataset and save them to a gz file.
    
    Args:
        year (str): The year of data to download (e.g., "2024")
        num_docs (int): Number of documents to collect (default: 40000)
    """
    print(f"Loading CC News dataset for year {year}...")
    
    # Load dataset in streaming mode
    dataset = load_dataset("stanford-oval/ccnews", name=year, streaming=True)
    
    # Collect documents
    documents = []
    print("Collecting documents...")
    for item in tqdm(dataset["train"], desc="Processing", unit=" docs"):
        if len(documents) >= num_docs:
            break
            
        # Extract text content, clean and validate
        text = item.get('text', '').strip()
        if text and len(text.split()) > 10:  # Basic validation: at least 10 words
            documents.append(text)
    
    # Ensure we have enough documents
    if len(documents) < num_docs:
        raise ValueError(f"Only found {len(documents)} documents, needed {num_docs}")
    
    # Randomly sample if we have more than needed
    if len(documents) > num_docs:
        documents = random.sample(documents, num_docs)
    
    # Save to gzip file
    output_file = "ccnews_documents.gz"
    print(f"Saving {len(documents)} documents to {output_file}...")
    
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n')
    
    print("Done! You can now use this file with the FastText service.")
    print(f"Total documents saved: {len(documents)}")

if __name__ == "__main__":
    try:
        download_and_prepare_documents()
    except Exception as e:
        print(f"Error: {str(e)}")