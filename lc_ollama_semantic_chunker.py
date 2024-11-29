import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings


breakpoint_threshold_type = "percentile"  # Options: "standard_deviation", "interquartile"

# Create the SemanticChunker with Ollama Embeddings
text_splitter = SemanticChunker(
    embeddings=OllamaEmbeddings(model="llama3.2"), breakpoint_threshold_type=breakpoint_threshold_type
)

text = """Penguins are flightless, aquatic birds that are native to the Southern Hemisphere. They are known for their distinctive black and white plumage, which provides camouflage both in the water and on land. Penguins are highly social animals, living in large colonies that can number in the thousands. 
They are excellent swimmers and divers, using their flipper-like wings to propel themselves through the water. Penguins feed primarily on fish, squid, and krill, which they catch by diving deep into the ocean. They are also known for their unique mating rituals, which often involve elaborate displays of courtship. 
Penguins are facing a number of threats, including climate change, habitat loss, and overfishing. However, conservation efforts are underway to protect these fascinating creatures. Their waddling gait on land and graceful movements in the water make them a delight to observe. 
Penguins are a true testament to the diversity and wonder of the natural world."""

documents = text_splitter.create_documents([text])

print(documents)