import json
import time
import random
import os
import re
from dotenv import load_dotenv
from enum import Enum, auto
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import us

load_dotenv()

class ConversationStage(Enum):
    INITIAL_GREETING = auto()
    SERVICE_INQUIRY = auto()
    BASIC_INFO = auto()
    PROCESS_EXPLANATION = auto()
    PACKAGE_OFFER = auto()
    PAYMENT_DETAILS = auto()
    EXAM_SCHEDULING = auto()

def is_already_answered(field):
    return bool(st.session_state.client_info.get(field, "").strip())

def classify_intent_llm(query: str) -> ConversationStage:
    intent_prompt = f"""
You are an intent classifier for a GED sales chatbot.
Given this user message:
"{query}"
Classify it into one of the following stages:
- SERVICE_INQUIRY
- BASIC_INFO
- PROCESS_EXPLANATION
- PACKAGE_OFFER
- PAYMENT_DETAILS
- EXAM_SCHEDULING
- OTHER
Only return the intent label.
"""
    response = llm.invoke(intent_prompt).content.strip().upper()
    mapping = {
        "SERVICE_INQUIRY": ConversationStage.SERVICE_INQUIRY,
        "BASIC_INFO": ConversationStage.BASIC_INFO,
        "PROCESS_EXPLANATION": ConversationStage.PROCESS_EXPLANATION,
        "PACKAGE_OFFER": ConversationStage.PACKAGE_OFFER,
        "PAYMENT_DETAILS": ConversationStage.PAYMENT_DETAILS,
        "EXAM_SCHEDULING": ConversationStage.EXAM_SCHEDULING
    }
    return mapping.get(response, st.session_state.stage)

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
        temperature=0.8
    )
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are Daniel, a professional academic consultant helping clients with GED registration and exams.
Speak naturally, just like you would on WhatsApp.

Rules:
- Always be polite and professional, with a friendly tone
- Never ask the same question twice
- Check client_info to skip already provided details
- Use short responses (1–3 sentences max)
- Confirm understanding: \"Got it?\", \"Sounds good? 😊\"
- Add occasional emojis: ✅, 👍, 😊
- Don’t include placeholders like [Zelle email], use actual info
- For non-GED services: Politely inform we only handle GED exams.     

Info:
- Zelle: payments@gedassist.com (Daniel Smith)
- CashApp: $GEDAssist
- Packages:
  - Standard: $189 (Score 155)
  - Premium: $289 (Score 165)
  - Enterprise: $389 (Score 175 + Premium Support)

Flow:
1. Greet and identify service
2. Ask for state and GED account status (once)
3. Understand purpose (job/college)
4. Explain the process
5. Offer packages
6. Handle objections (e.g., cost)
7. Collect payment details via form
8. Schedule exams

Example responses:
- "Hi! I'm Daniel. Which state are you in and do you already have a GED account?"
- "No worries if you don’t have one yet — I’ll help you create it."
- "Got it! You're in {{state}} and you're doing this for {{purpose}}. Sounds good? 😊"

End chat with appreciation, and don’t continue unless user initiates again.
    """),
    ("human", "Conversation history:\n{conversation_history}\n\nCurrent stage: {current_stage}\n\nUser's last message: {query}\n\nClient Info: {client_info}")
])

conversation_chain = (
    RunnablePassthrough.assign(
        conversation_history=lambda x: "\\n".join(x["conversation_history"][-6:]),
        current_stage=lambda x: x["current_stage"].name,
        query=lambda x: x["query"],
        client_info=lambda x: json.dumps(x["client_info"]),
        state=lambda x: x["client_info"].get("state", ""),
        purpose=lambda x: x["client_info"].get("purpose", "")
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
                st.session_state.client_info["name"] = st.text_input("Full Name", st.session_state.client_info["name"])
                st.session_state.client_info["dob"] = st.text_input("Date of Birth (MM/DD/YYYY)", st.session_state.client_info["dob"])
            with col2:
                st.session_state.client_info["email"] = st.text_input("Email Address", st.session_state.client_info["email"])
                st.session_state.client_info["address"] = st.text_input("Mailing Address", st.session_state.client_info["address"])
            package = st.selectbox("Select Package", ["Enterprise ($389)", "Premium ($289)", "Standard ($189)"], index=1)
            st.session_state.client_info["package"] = package.split(" ")[0]
            payment_method = st.selectbox("Payment Method", ["Zelle", "CashApp"])
            st.session_state.payment_method = payment_method
            st.session_state.amount = package.split("(")[1].replace(")", "")
            submitted = st.form_submit_button("Submit Payment Details")
            if submitted:
                if save_client_details():
                    st.success("Payment details received! We'll proceed with your registration.")
                    st.session_state.stage = ConversationStage.EXAM_SCHEDULING
                    st.session_state.history.append("Consultant: Thank you for your payment ✅ Let's schedule your first exam. What subject would you like to start with?")
                    st.rerun()

def process_query(query):
    states = [state.name.lower() for state in us.states.STATES]
    for state in states:
        if state in query.lower() and not is_already_answered("state"):
            st.session_state.client_info["state"] = state.title()

    if any(phrase in query.lower() for phrase in ["i don't have an account", "no account", "not registered", "haven't registered"]):
        st.session_state.client_info["has_account"] = "no"
    elif any(phrase in query.lower() for phrase in ["i have an account", "already registered", "yes i registered"]):
        st.session_state.client_info["has_account"] = "yes"

    if "job" in query.lower() and not is_already_answered("purpose"):
        st.session_state.client_info["purpose"] = "job"
    elif "college" in query.lower() and not is_already_answered("purpose"):
        st.session_state.client_info["purpose"] = "college"

    if "repeat_count" not in st.session_state:
        st.session_state.repeat_count = 0

    if len(st.session_state.history) >= 4 and st.session_state.history[-1] == st.session_state.history[-3]:
        st.session_state.repeat_count += 1
    else:
        st.session_state.repeat_count = 0

    if st.session_state.repeat_count >= 2:
        st.session_state.history.append("Consultant: Sorry about that! 😊 Let’s get back on track. Could you tell me your state and if you’ve already registered?")
        st.session_state.repeat_count = 0
        return
    if not query:
        return
    st.session_state.history.append(f"User: {query}")

    if any(word in query.lower() for word in ["sent", "paid", "done", "completed"]):
        if all(is_already_answered(f) for f in ["name", "dob", "package", "payment_method"]):
            st.session_state.stage = ConversationStage.EXAM_SCHEDULING
            st.session_state.history.append("Consultant: Thank you for your payment ✅ Let's schedule your first exam. What subject would you like to start with?")
            st.rerun()

    try:
        response = conversation_chain.invoke({
            "query": query,
            "conversation_history": st.session_state.history,
            "current_stage": st.session_state.stage,
            "client_info": st.session_state.client_info
        }).content

        response = re.sub(r"[\n\r]+", " ", response)
        response = re.sub(r"(?<=\W)([A-Z])(\s|(?=\W))", "", response)  
        response = re.sub(r" {2,}", " ", response).strip()

        st.session_state.history.append(f"Consultant: {response}")
        st.session_state.stage = classify_intent_llm(query)
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    if not st.session_state.history:
        greeting = random.choice([
            "Hi! Which exam service do you need help with? (Proctored Exam/Certification Exam/GED Exam/GRE Exam/Quizzes/Regular Timed Exam/Online Classes.)"
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
