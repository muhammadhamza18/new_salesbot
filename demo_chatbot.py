import json
import time
import random
import os
from dotenv import load_dotenv
from enum import Enum, auto
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class ConversationStage(Enum):
    INITIAL_GREETING = auto()
    SERVICE_INQUIRY = auto()
    BASIC_INFO = auto()
    PROCESS_EXPLANATION = auto()
    PACKAGE_OFFER = auto()
    PAYMENT_DETAILS = auto()
    EXAM_SCHEDULING = auto()

def classify_intent(query: str) -> ConversationStage:
    query = query.lower()
    if any(word in query for word in ["ged", "gre", "exam", "service"]):
        return ConversationStage.SERVICE_INQUIRY
    elif any(word in query for word in ["florida", "job", "college", "state", "account"]):
        return ConversationStage.BASIC_INFO
    elif "how does" in query or "process" in query or "explain" in query:
        return ConversationStage.PROCESS_EXPLANATION
    elif any(word in query for word in ["package", "price", "cost"]):
        return ConversationStage.PACKAGE_OFFER
    elif any(word in query for word in ["zelle", "cashapp", "pay", "payment"]):
        return ConversationStage.PAYMENT_DETAILS
    elif any(word in query for word in ["schedule", "test", "exam date"]):
        return ConversationStage.EXAM_SCHEDULING
    return st.session_state.stage

if "history" not in st.session_state:
    st.session_state.history = []
if "stage" not in st.session_state:
    st.session_state.stage = ConversationStage.INITIAL_GREETING
if "client_info" not in st.session_state:
    st.session_state.client_info = {
        "service": "",
        "state": "",
        "purpose": "",
        "has_account": "",
        "name": "",
        "dob": "",
        "address": "",
        "email": "",
        "package": ""
    }
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = set()

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
    ("system", """
You are Daniel, a professional academic consultant who helps students with GED and other exam services. Follow these guidelines:

1. Be concise - responses should be 1-3 sentences max
2. Never repeat questions already asked/answered
3. Keep conversation flowing naturally
4. Only ask for information not already provided
5. Don't include placeholders like [Your Zelle Email] - use actual details

Client Info: {client_info}

Key Process Flow:
1. Identify service needed (GED/GRE/etc)
2. Ask for state and registration status (once)
3. Explain process briefly
4. Offer package options
5. Collect payment details via form
6. Schedule exams

Payment Details:
- Zelle: payments@gedassist.com (Daniel Smith)
- CashApp: $GEDAssist

Packages:
- Standard: $189 (basic)
- Premium: $289 (recommended)
- Enterprise: $389 (premium support)

Always:
- Be professional but friendly
- Confirm understanding ("Got it?", "Sounds Good?")
- Use occasional emojis (👍, ✅)
- Move conversation forward
- Never repeat yourself
"""),
    ("human", "Conversation history:\n{conversation_history}\n\nCurrent stage: {current_stage}\n\nUser's last message: {query}")
])

conversation_chain = (
    RunnablePassthrough.assign(
        conversation_history=lambda x: "\n".join(x["conversation_history"][-6:]),
        current_stage=lambda x: x["current_stage"].name,
        query=lambda x: x["query"],
        client_info=lambda x: json.dumps(x["client_info"])
    )
    | prompt_template
    | llm
)

def save_client_details():
    try:
        data = {
            "name": st.session_state.client_info["name"],
            "package": st.session_state.client_info["package"],
            "payment_method": st.session_state.payment_method,
            "amount": st.session_state.amount,
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
    st.title("GED Exam Assistance")

    for i, message in enumerate(st.session_state.history):
        role = "assistant" if i % 2 == 0 else "user"
        with st.chat_message(role):
            st.write(message)

    if st.session_state.stage == ConversationStage.PAYMENT_DETAILS:
        with st.form("payment_form"):
            st.subheader("Complete Your Registration")

            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.client_info["name"] == "":
                    st.session_state.client_info["name"] = st.text_input("Full Name")
                if st.session_state.client_info["dob"] == "":
                    st.session_state.client_info["dob"] = st.text_input("Date of Birth (MM/DD/YYYY)")
            with col2:
                if st.session_state.client_info["email"] == "":
                    st.session_state.client_info["email"] = st.text_input("Email Address")
                if st.session_state.client_info["address"] == "":
                    st.session_state.client_info["address"] = st.text_input("Mailing Address")

            package = st.selectbox("Select Package", ["Enterprise ($389)", "Premium ($289)", "Standard ($189)"], index=1)
            st.session_state.client_info["package"] = package.split(" ")[0]

            payment_method = st.selectbox("Payment Method", ["Zelle", "CashApp"])
            st.session_state.payment_method = payment_method

            amount = package.split("(")[1].replace(")", "")
            st.session_state.amount = amount

            submitted = st.form_submit_button("Submit Payment Details")
            if submitted:
                if save_client_details():
                    st.success("Payment details received! We'll proceed with your registration.")
                    st.session_state.stage = ConversationStage.EXAM_SCHEDULING
                    st.session_state.history.append("Consultant: Thank you for your payment. When would you like to schedule your first exam?")
                    st.rerun()

def process_query(query):
    if not query:
        return

    try:
        st.session_state.history.append(f"User: {query}")

        response = conversation_chain.invoke({
            "query": query,
            "conversation_history": st.session_state.history,
            "current_stage": st.session_state.stage,
            "client_info": st.session_state.client_info
        }).content

        response = "\n".join([line for line in response.split("\n") if line.strip() and not line.strip().startswith("[")])
        st.session_state.history.append(f"Consultant: {response}")
        st.session_state.stage = classify_intent(query)
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    if not st.session_state.history:
        greeting = random.choice([
            "Hi! Which exam service do you need help with? (Proctored Exam/Certification Exam/GED Exam/GRE Exam/Quizzes/Regular Timed Exam/Online Classes.)",
            # "Hello! Are you looking for GED exam assistance?",
            # "Welcome! How can I help you with your exams today?"
        ])
        st.session_state.history.append(f"Consultant: {greeting}")
        st.session_state.stage = ConversationStage.SERVICE_INQUIRY

    display_chat()

    if st.session_state.stage != ConversationStage.PAYMENT_DETAILS:
        query = st.chat_input("Type your message here...")
        if query:
            process_query(query)

if __name__ == "__main__":
    main()
