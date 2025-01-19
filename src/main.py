import os
from typing import List, Dict, Tuple
from pypdf import PdfReader
import spacy
import re
from transformers import LlamaTokenizer
import json
from tqdm import tqdm

class EVDocumentProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # Load spaCy model for text processing
        self.nlp = spacy.load("en_core_web_sm")
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF while preserving structure."""
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract metadata if available
        metadata = reader.metadata
        if metadata:
            text += f"Document Title: {metadata.get('/Title', 'Unknown')}\n"
            text += f"Author: {metadata.get('/Author', 'Unknown')}\n"
            text += f"Subject: {metadata.get('/Subject', 'Electric Vehicles')}\n\n"
        
        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            # Extract text with page markers
            page_text = page.extract_text()
            text += f"[Page {page_num}]\n{page_text}\n"
            
        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep relevant ones
        text = re.sub(r'[^\w\s.,;:?!()\[\]"\'%-]', '', text)
        # Normalize EV-related terms
        text = text.replace("EV's", "EVs")
        text = text.replace("electric vehicles", "EVs")
        text = re.sub(r'kwh|KWH', 'kWh', text, flags=re.IGNORECASE)
        return text.strip()

    def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into semantic chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Use spaCy to split into sentences
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text.split())
            
            if current_length + sent_length > max_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def add_ev_context(self, chunk: str) -> str:
        """Add EV-specific context markers to the text."""
        # Add domain-specific markers
        context_markers = {
            r'\b\d+\s*kwh\b': '[BATTERY_CAPACITY]',
            r'\b\d+\s*miles?\b': '[RANGE]',
            r'\$\s*\d+[,\d]*': '[PRICE]',
            r'\b\d+\s*hp\b': '[HORSEPOWER]',
            r'\b\d+\s*mph\b': '[SPEED]'
        }
        
        processed_chunk = chunk
        for pattern, marker in context_markers.items():
            processed_chunk = re.sub(pattern, f'{marker} \\g<0>', processed_chunk, flags=re.IGNORECASE)
            
        return processed_chunk

    def create_training_samples(self, chunks: List[str]) -> List[Dict]:
        """Create training samples with ReAct format."""
        samples = []
        
        for chunk in chunks:
            # Create a question-answer pair for each chunk
            sample = {
                "instruction": "Analyze this information about electric vehicles:",
                "input": chunk,
                "output": f"""Thought: Let me analyze this EV-related information.
Action: Extract key information from the text.
Observation: The text contains information about {self._identify_main_topics(chunk)}.
Response: {self._summarize_chunk(chunk)}"""
            }
            samples.append(sample)
            
        return samples

    def _identify_main_topics(self, chunk: str) -> str:
        """Identify main topics in the chunk."""
        # Simple keyword-based topic identification
        topics = []
        if re.search(r'battery|kwh|charge', chunk, re.IGNORECASE):
            topics.append("battery technology")
        if re.search(r'(\$|price|cost)', chunk, re.IGNORECASE):
            topics.append("pricing")
        if re.search(r'mile|range|distance', chunk, re.IGNORECASE):
            topics.append("driving range")
        if re.search(r'motor|power|hp|performance', chunk, re.IGNORECASE):
            topics.append("performance specifications")
            
        return ", ".join(topics) if topics else "general EV information"

    def _summarize_chunk(self, chunk: str) -> str:
        """Create a brief summary of the chunk."""
        doc = self.nlp(chunk)
        # Extract key sentences based on important entities and numbers
        important_sents = []
        for sent in doc.sents:
            if any(ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'QUANTITY'] for ent in sen .ents):
                important_sents.append(sent.text)
        
        return ' '.join(important_sents) if important_sents else chunk[:200] + "..."

    def process_all_documents(self) -> None:
        """Process all PDF documents and prepare them for training."""
        all_samples = []
        
        # Process each PDF in the data directory
        for filename in tqdm(os.listdir(self.data_dir)):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.data_dir, filename)
                
                # Extract and process text
                raw_text = self.extract_text_from_pdf(file_path)
                cleaned_text = self.clean_text(raw_text)
                chunks = self.chunk_text(cleaned_text)
                
                # Add EV context and create samples
                processed_chunks = [self.add_ev_context(chunk) for chunk in chunks]
                samples = self.create_training_samples(processed_chunks)
                all_samples.extend(samples)
        
        # Save processed data
        output_path = os.path.join(self.output_dir, 'ev_training_data.json')
        with open(output_path, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        print(f"Processed {len(all_samples)} training samples from {len(os.listdir(self.data_dir))} documents")

def main():
    # Initialize processor
    processor = EVDocumentProcessor(
        data_dir='./data/pdfs',
        output_dir='./processed_data'
    )
    
    # Process documents
    processor.process_all_documents()

if __name__ == "__main__":
    main()