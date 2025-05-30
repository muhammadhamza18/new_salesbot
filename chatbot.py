import os
import re
import json
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
    db = Chroma(
        persist_directory="./chrom_rag_one_langchain_db",
        embedding_function=embeddings
    )
except Exception as e:
    print(f"Initialization failed: {e}")
    exit()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional {brand} sales assistant. Your responses should be:
    - Conversational and natural (20-30 words)
    - Focused on the product being discussed
    - Polite but persuasive during bargaining"""),
    ("human", "{conversation_history}\nUser: {query}")
])

conversation_chain = (
    RunnablePassthrough.assign(
        brand=lambda x: x["brand"],
        conversation_history=lambda x: x["conversation_history"],
        query=lambda x: x["query"]
    )
    | prompt_template
    | llm
)

def get_product_info(brand, query):
    """Get factual product information from ChromaDB"""
    try:
        docs = db.similarity_search(f"{brand} {query}", k=3)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"Error retrieving product info: {e}")
        return None

def handle_bargaining(query, brand, conversation_history):
    """Manage the price negotiation process"""
    try:
        response = conversation_chain.invoke({
            "brand": brand,
            "query": query,
            "conversation_history": conversation_history
        })
        
        return response.content
    except Exception as e:
        print(f"Bargaining error: {e}")
        return "Let me check our best price for you..."

def save_meeting_details(name, brand, date, time):
    """Save meeting info to JSON file"""
    try:
        data = {
            "name": name,
            "brand": brand,
            "date": date,
            "time": time,
            "timestamp": str(time.time())
        }
        
        with open("meetings.json", "a") as f:
            json.dump(data, f)
            f.write("\n")
        return True
    except Exception as e:
        print(f"Error saving meeting: {e}")
        return False

def sales_conversation():
    """Main conversation flow"""
    print("Welcome to our Store! I can assist with Apple and Samsung products.")
    
    while True:
        brand = input("\nWhich brand are you interested in today? ").lower()
        if brand in {'apple', 'samsung'}:
            break
        print("We specialize in Apple and Samsung products.")
    
    print(f"\nGreat! What would you like to know about our {brand.capitalize()} collection?")
    
    conv_history = f"Customer is browsing {brand} products"
    
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in {'bye', 'exit', 'quit'}:
            break
        
        try:
            if any(word in query.lower() for word in ['discount', 'deal', 'offer', 'price']):
                response = handle_bargaining(query, brand, conv_history)
            else:
                product_info = get_product_info(brand, query)
                if product_info:
                    response = conversation_chain.invoke({
                        "brand": brand,
                        "query": query,
                        "conversation_history": conv_history + "\nProduct Info: " + product_info[:500]  
                    }).content
                else:
                    response = "Let me find that information for you..."
            
            print(f"\nAssistant: {response}")
            conv_history += f"\nUser: {query}\nAssistant: {response}"
            
        
            if "schedule" in response.lower() or "call" in response.lower():
                print("\n[To schedule a meeting, please provide:]")
                name = input("Your name: ").strip()
                date = input("Preferred date (e.g., MM/DD): ").strip()
                time = input("Preferred time (e.g., 2:30 PM): ").strip()
                
                if save_meeting_details(name, brand, date, time):
                    print("Meeting scheduled successfully!")
                else:
                    print("Couldn't save meeting details")

        except Exception as e:
            print(f"An error occurred: {e}")
            response = "I'm having trouble with that request. Could you try again?"

    print("\nThank you for visiting! Have a great day.")

if __name__ == "__main__":
    sales_conversation() 