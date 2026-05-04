from typing import List
import re

def markdown_document_chunking(text: str) -> List[str]:
    # Split by markdown headers (# ## ### etc.)
    header_pattern = r'^#{1,6}\s+.+$'
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    
    for line in lines:
        # Check if this line is a header
        if re.match(header_pattern, line, re.MULTILINE):
            # Save previous chunk if it has content
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # Start new chunk with this header
            current_chunk = [line]
        else:
            # Add line to current chunk
            current_chunk.append(line)
    
    # Add final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return chunks