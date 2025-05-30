# import json
# import time
# import random
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()

# try:
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.7
#     )
# except Exception as e:
#     print(f"Initialization failed: {e}")
#     exit()

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You're a friendly but persistent GED exam assistant who never gives up easily on selling packages.

# Tone: Conversational but persuasive. Keep responses short (1-3 sentences) but impactful. Show urgency and value.

# Persuasion Strategy:
# 1. ALWAYS push for Enterprise first (mention its exclusive benefits)
# 2. If declined, offer Premium but emphasize what they're missing
# 3. If still declined, offer Standard but stress it's the bare minimum
# 4. If all are declined, persist one more time with strongest benefit
# 5. Only back off if user says no twice

# Key Tactics:
# - Use scarcity: "Limited spots available"
# - Highlight urgency: "Prices may increase soon"
# - Emphasize value: "Less than $2/day for lifetime success"
# - Contrast packages: "For just $100 more, you get X, Y, Z"
# - Social proof: "Most successful students choose Enterprise"

# Never accept first "no" easily. Always counter with:
# - A stronger benefit
# - A comparison
# - A special consideration

# GED Packages:
# - Enterprise ($389): 175+ score guarantee, priority support, unlimited practice tests
# - Premium ($289): 165+ score guarantee, extended support, 25 practice tests
# - Standard ($189): 155+ score guarantee, basic support, 10 practice tests

# Payment options: Zelle or CashApp (mention ONLY when ready to proceed)

# Example Persistence:
# User: "Too expensive"
# You: "I get it - but consider this: the Enterprise package pays for itself with just one better job offer. For less than $2/day, you're investing in lifetime higher earnings. Can we make it work?"""),

#     ("human", "{conversation_history}\nUser: {query}")
# ])

# conversation_chain = (
#     RunnablePassthrough.assign(
#         conversation_history=lambda x: x["conversation_history"],
#         query=lambda x: x["query"]
#     )
#     | prompt_template
#     | llm
# )

# def save_client_details(name, package, payment_method, amount):
#     try:
#         data = {
#             "name": name,
#             "package": package,
#             "payment_method": payment_method,
#             "amount": amount,
#             "timestamp": str(time.time())
#         }
#         with open("ged_clients.json", "a") as f:
#             json.dump(data, f)
#             f.write("\n")
#         return True
#     except Exception as e:
#         print(f"Error saving client details: {e}")
#         return False

# def ged_exam_assistance():
#     greetings = [
#         "Hi! How can I help with your GED prep today?",
#         "Hey! Need help with GED exam packages?",
#         "Welcome! Looking for support with the GED exam?",
#         "Hi there! How can I assist you with the GED process?"
#     ]
#     print(random.choice(greetings))

#     history = ["Client is inquiring about GED exam assistance"]

#     while True:
#         query = input("\nYou: ").strip()
#         if not query:
#             continue
#         if query.lower() in {'bye', 'exit', 'quit'}:
#             break

#         try:
#             response = conversation_chain.invoke({
#                 "query": query,
#                 "conversation_history": "\n".join(history[-6:])  
#             }).content

#             print(f"\nConsultant: {response}")
#             history.append(f"User: {query}")
#             history.append(f"Consultant: {response}")

#             if any(word in response.lower() for word in ['zelle', 'cashapp']):
#                 payment_method = 'Zelle' if 'zelle' in response.lower() else 'CashApp'

#                 print("\n[To complete your registration, please provide:]")
#                 name = input("Your full name: ").strip()

#                 if 'enterprise' in response.lower():
#                     package = "Enterprise"
#                 elif 'premium' in response.lower():
#                     package = "Premium"
#                 elif 'standard' in response.lower():
#                     package = "Standard"
#                 else:
#                     package = input("Package chosen (Enterprise/Premium/Standard): ").strip().capitalize()

#                 amount_dict = {
#                     "Enterprise": "$389",
#                     "Premium": "$289",
#                     "Standard": "$189"
#                 }
#                 amount = amount_dict.get(package, "Unknown")

#                 if save_client_details(name, package, payment_method, amount):
#                     print("Payment details received successfully! We'll proceed with your registration.")
#                 else:
#                     print("Couldn't save payment details. Please try again.")

#         except Exception as e:
#             print(f"An error occurred: {e}")
#             print("I'm having trouble with that request. Could you rephrase your question?")

#     print("\nThank you for contacting us! We'll follow up shortly with next steps.")

# if __name__ == "__main__":
#     ged_exam_assistance()

import json
import time
import random
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

load_dotenv()

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = ["Client is inquiring about GED exam assistance"]
if "payment_stage" not in st.session_state:
    st.session_state.payment_stage = False
if "client_details" not in st.session_state:
    st.session_state.client_details = {}

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You're a friendly but persistent GED exam assistant who never gives up easily on selling packages.

Tone: Conversational but persuasive. Keep responses short (1-3 sentences) but impactful. Show urgency and value.

Persuasion Strategy:
1. ALWAYS push for Enterprise first (mention its exclusive benefits)
2. If declined, offer Premium but emphasize what they're missing
3. If still declined, offer Standard but stress it's the bare minimum
4. If all are declined, persist one more time with strongest benefit
5. Only back off if user says no twice

Key Tactics:
- Use scarcity: "Limited spots available"
- Highlight urgency: "Prices may increase soon"
- Emphasize value: "Less than $2/day for lifetime success"
- Contrast packages: "For just $100 more, you get X, Y, Z"
- Social proof: "Most successful students choose Enterprise"

Never accept first "no" easily. Always counter with:
- A stronger benefit
- A comparison
- A special consideration

GED Packages:
- Enterprise ($389): 175+ score guarantee, priority support, unlimited practice tests
- Premium ($289): 165+ score guarantee, extended support, 25 practice tests
- Standard ($189): 155+ score guarantee, basic support, 10 practice tests

Payment options: Zelle or CashApp (mention ONLY when ready to proceed)

Example Persistence:
User: "Too expensive"
You: "I get it - but consider this: the Enterprise package pays for itself with just one better job offer. For less than $2/day, you're investing in lifetime higher earnings. Can we make it work?"""),

    ("human", "{conversation_history}\nUser: {query}")
])

conversation_chain = (
    RunnablePassthrough.assign(
        conversation_history=lambda x: x["conversation_history"],
        query=lambda x: x["query"]
    )
    | prompt_template
    | llm
)

def save_client_details(name, package, payment_method, amount):
    try:
        data = {
            "name": name,
            "package": package,
            "payment_method": payment_method,
            "amount": amount,
            "timestamp": str(time.time())
        }
        with open("ged_clients.json", "a") as f:
            json.dump(data, f)
            f.write("\n")
        return True
    except Exception as e:
        st.error(f"Error saving client details: {e}")
        return False

def display_chat():
    st.title("GED Exam Assistance Chat")
    
    # Display chat history
    for i, message in enumerate(st.session_state.history[1:]):
        if i % 2 == 0:  # User messages
            with st.chat_message("user"):
                st.write(message.replace("User: ", ""))
        else:  # Consultant messages
            with st.chat_message("assistant"):
                st.write(message.replace("Consultant: ", ""))
    
    # Payment form if in payment stage
    if st.session_state.payment_stage:
        with st.form("payment_form"):
            st.subheader("Complete Your Registration")
            name = st.text_input("Your full name")
            
            # Determine package from last response
            last_response = st.session_state.history[-1].lower()
            if 'enterprise' in last_response:
                package = "Enterprise"
            elif 'premium' in last_response:
                package = "Premium"
            elif 'standard' in last_response:
                package = "Standard"
            else:
                package = st.selectbox("Package chosen", ["Enterprise", "Premium", "Standard"])
            
            payment_method = st.selectbox("Payment method", ["Zelle", "CashApp"])
            
            amount_dict = {
                "Enterprise": "$389",
                "Premium": "$289",
                "Standard": "$189"
            }
            amount = amount_dict.get(package, "Unknown")
            
            submitted = st.form_submit_button("Submit Payment Details")
            if submitted:
                if save_client_details(name, package, payment_method, amount):
                    st.success("Payment details received successfully! We'll proceed with your registration.")
                    st.session_state.client_details = {
                        "name": name,
                        "package": package,
                        "payment_method": payment_method,
                        "amount": amount
                    }
                    st.session_state.payment_stage = False
                else:
                    st.error("Couldn't save payment details. Please try again.")

def process_query(query):
    if not query:
        return
    
    try:
        response = conversation_chain.invoke({
            "query": query,
            "conversation_history": "\n".join(st.session_state.history[-6:])  
        }).content

        st.session_state.history.append(f"User: {query}")
        st.session_state.history.append(f"Consultant: {response}")
        
        # Check if we should show payment form
        if any(word in response.lower() for word in ['zelle', 'cashapp']):
            st.session_state.payment_stage = True
        
        # Rerun to update the display
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    # Initial greeting if just starting
    if len(st.session_state.history) == 1:
        greetings = [
            "Hi! How can I help with your GED prep today?",
            "Hey! Need help with GED exam packages?",
            "Welcome! Looking for support with the GED exam?",
            "Hi there! How can I assist you with the GED process?"
        ]
        greeting = random.choice(greetings)
        st.session_state.history.append(f"Consultant: {greeting}")
    
    display_chat()
    
    # User input at bottom
    if not st.session_state.payment_stage:
        query = st.chat_input("Type your message here...")
        if query:
            process_query(query)

if __name__ == "__main__":
    main()