# Terraform RAG System

The Terraform RAG System was created to help cloud-engineers with the deployment of IaC


## Project Structure
The folder structure for this project was planned and implemented using Claude Code. Placeholder files have been added to some directories and currently contain no code. 

## The Knowledge Base
For this RAG system to be effective at giving trustworthy responses, I will need to source the documentation from HashiCorp's official docs, [Github repo](https://github.com/hashicorp/web-unified-docs).

To start, I will be cloning only the `web-unified-docs/content/terraform/v1.14.x` folder, which should provide sufficient documentation for the chatbot to produce meaningful responses regarding the CLI, syntax, and other basic questions. 

Additional documentation will be added to the knowledge base once gaps are identified. 

## The Ingestion Pipeline

This section will cover the ingestion pipeline of this RAG system. We will go through document loading using the docs [here](./data/docs/), chunking, vector embeddings using an embedding model, then storage in a vector database. 

### The Document Loader

The document loader [`loader.py`](./src/terraform_rag/ingestion/loader.py) will be used to split the docs content and metadata and prepare them for chunking. Since HashiCorps documentation is already written in markdown, no cleanup will be needed at this time. The only concern are the occasional .jsx tags, but that will be addressed if there is a noticeable issue with quality with responses. 


