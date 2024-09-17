import os
import gradio as gr
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAI
from docx import Document
import PyPDF2
import os
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import nest_asyncio
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ['PINECONE_API_KEY'] = "c5207cae-fc27-493e-a508-2292a1659659"

# Creating the index for Pinecone
pc = Pinecone(api_key="c5207cae-fc27-493e-a508-2292a1659659")

# Check if the index exists and delete it if necessary
try:
    pc.delete_index(name="example-index")
    print("Existing index deleted.")
except Exception as e:
    print(f"Error deleting index: {e}")

pc.create_index(
    name="example-index",
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    deletion_protection="disabled"
)

# Function to read the content of a Word file
def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

# Function to read the content of a PDF file
def read_pdf_file(file_path):
    pdf_text = []
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text.append(page.extract_text())
    return "\n".join(pdf_text)

# Function to load the document based on its type (PDF or Word)
def load_document(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == ".docx":
        return read_word_file(file_path)
    elif file_extension == ".pdf":
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type. Only .docx and .pdf are supported.")
    
def embed_document(document_content):
      # Chunk the document based on h2 headers.
      markdown_document = document_content

      # Define the headers to split the document on;
      headers_to_split_on = [("##", "Header 2")]

      # Pinecone API key for authentication
      pinecone_api_key="c5207cae-fc27-493e-a508-2292a1659659"

      # It will split the document based on the provided headers
      markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
      md_header_splits = markdown_splitter.split_text(markdown_document)

      # Define the embedding model to be used
      model_name = "multilingual-e5-large"

      # Initialize the Pinecone embeddings with the chosen model and API key
      embeddings = PineconeEmbeddings(
                                      model=model_name,
                                      pinecone_api_key=pinecone_api_key
                                      )


      # Embed each chunk and upsert the embeddings into your Pinecone index.
      docsearch = PineconeVectorStore.from_documents(
                                      documents=md_header_splits,
                                      index_name="example-index",
                                      pinecone_api_key=pinecone_api_key,
                                      embedding=embeddings,
                                      namespace="wondervector5000"
                                      )

      # Add a short delay (1 second) to ensure all processes are completed before any further action
      time.sleep(1)


def ask(query):
     # Initialize the Pinecone embeddings with the chosen model and API key
    embeddings = PineconeEmbeddings(
                                      model="multilingual-e5-large",
                                      pinecone_api_key="c5207cae-fc27-493e-a508-2292a1659659"
                                      )
      
    # Retrieve information from the Pinecone vector index to provide context
    knowledge = PineconeVectorStore.from_existing_index(
        index_name="example-index",
        namespace="wondervector5000",
        embedding=embeddings
    )

    # Retrieve relevant document based on the query
    qa = knowledge.as_retriever()
    result = qa.invoke(query)

    if result:
        page_content = result[0].page_content
    input_text1 = page_content[:800]

    input_text="find the ans of ->"+query+"in the following text->"+input_text1

    # Load GPT-2 tokenizer and generative model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # The retrieved content from Pinecone is passed as input to GPT-2 to generate a more contextually accurate and reliable response.
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=280, num_beams=5, pad_token_id=tokenizer.eos_token_id, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Truncate the response at the last full stop
    last_period_index = generated_text.rfind('.')
    if last_period_index != -1:
        generated_text = generated_text[:last_period_index-1]

    # Print the query and the generated response
    return generated_text, input_text1



# Apply nest_asyncio to avoid event loop issues in Colab
nest_asyncio.apply()

# Define the function for question answering
def gradio_app(pdf_file, query):
    # Check if both a query and a file have been provided
    if query and pdf_file:
        document_content = load_document(pdf_file.name)
        docsearch = embed_document(document_content)
        response, retrieved  = ask(query)

        return response, retrieved
    else:
        return "Please upload a PDF and ask a question.", ""

# Gradio interface setup
def main():
    # Create the interface
    interface = gr.Interface(
        fn=gradio_app,
        inputs=[
            gr.File(label="Upload PDF", file_types=["pdf"]),  # File input for PDF
            gr.Textbox(label="Ask a question")  # Textbox input for question
        ],
        outputs=[
            gr.Textbox(label="Answer"),  # Output for the answer
            gr.Textbox(label="Retrieved Document Segment")  # Output for retrieved document content
        ],
        title="PDF Question Answering System",
        description="Upload a PDF document and ask any questions to retrieve contextually relevant answers."
    )
    # Launch the interface
    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)



# Run the app
if __name__ == "__main__":
    main()
