import os
import sys
from typing import Annotated, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Load Environment Variables
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Please set it in a .env file.")
    sys.exit(1)

# --- CONFIGURATION ---
FILE_PATH = "zakaria-ayad.txt"
DB_PATH = "./chroma_db"

# 2. Setup LLM and Embeddings (Free Tier)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def setup_vectorstore():
    """Reads the file, splits it, and creates a vector store."""
    print("--- Indexing Document ---")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Could not find {FILE_PATH}")

    loader = TextLoader(FILE_PATH, encoding="utf-8")
    docs = loader.load()

    # Split text into manageable chunks for the LLM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create/Persist Vector Store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="zakaria_collection",
        persist_directory=DB_PATH
    )
    return vectorstore

# Initialize Vector Store (In production, you'd only do this once)
vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LANGGRAPH SETUP ---

# 3. Define State
# The state holds the history of the conversation and context.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str

# 4. Define Nodes

def retrieve_node(state: AgentState):
    """Retrieves relevant documents based on the last user message."""
    print("--- Node: Retrieve ---")
    last_message = state["messages"][-1]
    query = last_message.content
    
    # Retrieve docs
    docs = retriever.invoke(query)
    
    # Combine doc content into a single string
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_node(state: AgentState):
    """Generates an answer using the retrieved context."""
    print("--- Node: Generate ---")
    context = state["context"]
    messages = state["messages"]
    
    # System prompt to enforce behavior
    system_prompt = (
        "You are a helpful assistant knowledgeable about Zakaria Ayad. "
        "Use the following context to answer the user's question. "
        "If the answer is not in the context, say you don't know."
        "\n\nContext:\n" + context
    )
    
    # Create a prompt chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": [response]}

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Bot is ready! Ask about Zakaria Ayad (type 'quit' to exit).")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # Stream the graph execution
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for output in app.stream(inputs):
            pass # We just wait for the final state, or we could print intermediate steps
            
        # The final output is in the 'generate' key of the last yielded dictionary,
        # but simpler to just print the last message from the final state.
        # However, app.invoke returns the final state directly.
        final_state = app.invoke(inputs)
        print(f"Bot: {final_state['messages'][-1].content}")