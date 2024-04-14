# Chat with Websites

**Chat with Websites** is a sophisticated chatbot application that leverages natural language processing (NLP) and retrieval-augmented generation to interact with and extract information from websites. This project uses Streamlit for the frontend and a combination of LangChain and OpenAI technologies for backend processing.

## Live Demo

Experience the chatbot in action here: [**Live Demo**](https://chat-with-websites-ub5miumjqbahqnhoh3pd5s.streamlit.app/)

## Screenshot

![Deployment Screenshot](Screenshot%202024-04-14%20225045.png)

## Application Overview

The application processes data in several steps:

1. **Website Text Extraction:** Uses `WebBaseLoader` from LangChain to load and extract text from a specified website URL.
2. **Document Splitting:** Splits the text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Vector Store Creation:** Converts document chunks into vector embeddings using `Chroma` and `OpenAIEmbeddings`, facilitating efficient information retrieval.
4. **Retrieval Chain:** Implements a retrieval chain that integrates document-based retrieval with conversational context to generate search queries that are relevant to the ongoing conversation.
5. **Conversation Handling:** Manages a conversation where user inputs are processed to retrieve and generate responses using a combination of the context retriever chain and a retrieval-augmented generation chain.

### Technologies Used

- **Streamlit:** For creating the web app interface.
- **LangChain:** For constructing the logical flow of conversation and document handling.
- **OpenAI API:** Powers the underlying language models and document embedding processes.

## Installation

To get this project up and running locally:

```bash
# Clone the repository
git clone https://github.com/himanshu9178/chat-with-websites.git
cd chat-with-websites

# Install required dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
## How It Works

- Users enter a website URL and their OpenAI API key via the Streamlit sidebar.
- The chatbot initializes by loading and processing the website’s content into a vector store.
- Users interact with the chatbot through a chat interface, where they can ask questions related to the website's content.
- The chatbot retrieves information and crafts responses based on the context of the entire conversation.


