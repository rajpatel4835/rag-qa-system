
# Retrieval-Augmented Generation (RAG) Model for QA Bot

## Overview
This project is a PDF/Word Question Answering bot that allows users to upload documents (PDF or DOCX) and ask questions about their content. The bot uses Pinecone and multilingual-e5-large model for document indexing and retrieval, and GPT-2 for generating contextually relevant answers.

The bot is deployed with a Gradio frontend interface, where users can easily interact with the system by uploading documents and asking questions. The backend uses Pinecone and multilingual-e5-large model for vector storage and OpenAI GPT-2 for answer generation.

## Features
- **Document Support**: Supports both PDF and DOCX files.
- **Vector Embeddings**: Uses Pinecone and multilingual-e5-large model for embedding documents and storing vectors.
- **QA Model**: Utilizes GPT-2 for generating answers based on the document content.
- **Interactive Interface**: Simple Gradio interface for uploading documents and asking questions.
- **Chunking for Large Documents**: Documents are split into smaller chunks for efficient embedding and retrieval.

## How to Use
### Uploading Files
1. Click on the **"Upload PDF"** button.
2. Select a **PDF** or **DOCX** file to upload.

### Asking Questions
1. Enter your question in the **textbox** under "Ask a question."
2. Press **Enter** or click the **Submit** button.

### Viewing the Response
- The bot will generate an answer based on the content of the document you uploaded.
- The response will be displayed in the **Answer** textbox.
- Additionally, the bot will display a **Retrieved Document Segment** showing the portion of the document that was most relevant to the question.

## Example Interaction
### Example 1
- **Document**: `example.pdf`
- **Question**: "What is the summary of the introduction?"
- **Answer**: "The introduction focuses on the key aspects of modern AI applications, highlighting..."
- **Retrieved Document Segment**: "Artificial intelligence has become a transformative force in..."

### Example 2
- **Document**: `report.docx`
- **Question**: "What are the future goals mentioned in the report?"
- **Answer**: "The report outlines the goals of scaling AI infrastructure by 2025..."
- **Retrieved Document Segment**: "Future goals include establishing an AI research center and..."

## Setup Instructions
To run this project locally using Docker, follow these steps:

### Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/)
- Clone this repository:
   ```bash
   git clone https://github.com/rajpatel4835/rag-qa-system
   cd rag-qa-bot
   ```

### Using Docker
1. **Build the Docker image**:
   ```bash
   docker build -t rag-qa-system .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 7860:7860 rag-qa-system
   ```

3. **Access the web interface**:
   - Open your browser and go to `http://localhost:7860`
   - Upload a document, ask questions, and see the bot in action!

## Handling Large Documents and Multiple Queries
The bot efficiently handles large documents by splitting them into chunks and embedding each chunk separately. It also processes multiple queries by maintaining efficient retrieval from Pinecone's vector index.

## Challenges and Solutions
### Handling Large Files
The use of document chunking ensures that even large documents can be embedded and processed efficiently without performance drops.

### Multiple Queries
The system supports multiple queries by ensuring that embeddings are retrieved from the Pinecone index for each query, and the relevant chunks of text are passed to GPT-2 for generating answers.

### Scaling
Using Pinecone for vector indexing allows for scalable solutions where large datasets of documents can be indexed and queried.

## Technologies Used
- **Python**: For backend logic and handling document processing.
- **Gradio**: For building the interactive user interface.
- **Pinecone**: For document embedding and vector search.
- **GPT-2**: For generating natural language responses to queries.
- **Docker**: For containerizing the application.
  
## Future Improvements
- **Support for additional document formats** like TXT or HTML.
- **Improved retrieval accuracy** by experimenting with different embeddings models.