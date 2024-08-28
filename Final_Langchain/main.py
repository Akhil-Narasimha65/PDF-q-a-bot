import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import re

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Function to process the PDF
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    pdf_reader = PyPDFLoader("temp.pdf")
    documents = pdf_reader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)

    return db

def extract_numbers_with_context(text):
    """Extract numbers along with their preceding words for context."""
    pattern = r'([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    return matches

def post_process_aggregation(question, answer):
    """Post-process the answer for aggregation questions with improved filtering."""
    lower_question = question.lower()
    
    # Extract numbers and context
    numbers_with_context = extract_numbers_with_context(answer)
    
    # Extract the filter condition from the question
    filter_words = ["region", "person", "division", "territory", "representative"]
    filter_condition = next((word for word in filter_words if word in lower_question), None)
    
    if filter_condition:
        filter_value_match = re.search(fr"{filter_condition}\s+(\w+)", lower_question, re.IGNORECASE)
        if filter_value_match:
            filter_value = filter_value_match.group(1).lower()
            filtered_numbers = [float(num) for context, num in numbers_with_context 
                                if filter_value in context.lower()]
            
            if filtered_numbers:
                total = sum(filtered_numbers)
                return f"The total sales for {filter_condition} '{filter_value.capitalize()}' is {total}. Details: {answer}"
            else:
                return f"No specific data found for {filter_condition} '{filter_value.capitalize()}'. Raw answer: {answer}"
    
    # If no filter condition or filtered results are found
    if numbers_with_context:
        total = sum(float(num) for _, num in numbers_with_context)
        return f"The total of all sales mentioned is {total}. This may not be specific to your query. Details: {answer}"
    
    return answer

# Initialize Streamlit app
st.title("PDF Q&A Bot (Using langchain model)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        db = process_pdf(uploaded_file)
    
    AGGREGATION_PROMPT = PromptTemplate.from_template("""
    Given the following conversation and a followup question, rephrase the followup question to be a standalone question.
    If the question requires any calculations or aggregations, please perform them and show your work.
    Make sure to filter the data based on any specified conditions (e.g., region, person, division, territory).
    Provide a step-by-step breakdown of your calculations.

    Chat History: {chat_history}
    Follow up Input: {question}

    Standalone question with calculations (if needed):
    """)

    llm = ChatOpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        condense_question_prompt=AGGREGATION_PROMPT,
        return_source_documents=True,
        verbose=False
    )

    chat_history = []

    st.header("Ask a question about your PDF")
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Getting answer..."):
                try:
                    result = qa({"question": user_question, "chat_history": chat_history})
                    processed_answer = post_process_aggregation(user_question, result['answer'])
                    
                    # Append the new question-answer pair to the chat history as a tuple
                    chat_history.append((user_question, processed_answer))
                    
                    st.write(f"**Question:** {user_question}")
                    st.write(f"**Answer:** {processed_answer}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a question.")
