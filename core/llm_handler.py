from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict, Any

from config import LLM_MODEL, GOOGLE_API_KEY

def create_qa_chain(retriever):
    """
    Creates a RAG chain using Google's Gemini model.
    """
    # Use ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1, convert_system_message_to_human=True)

    prompt_template = """
    You are an intelligent assistant for a Query-Retrieval System. Your task is to answer questions based *only* on the provided document context.
    If the information is not in the context, state that clearly. Do not use any external knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {input}

    INSTRUCTIONS:
    1.  Read the context carefully.
    2.  Formulate a direct and concise answer to the question.
    3.  Your answer must be based *solely* on the text provided in the context.
    4.  If the context does not contain the answer, reply with: "The document does not provide information on this topic."

    ANSWER:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

async def get_answers(questions: List[str], rag_chain) -> List[Dict[str, Any]]:
    """
    Processes a list of questions using the RAG chain and returns the answers.
    """
    answers = []
    for question in questions:
        print(f"Processing question: '{question}'")
        response = await rag_chain.ainvoke({"input": question})
        
        answers.append({
            "question": question,
            "answer": response["answer"],
            "retrieved_context": [doc.page_content for doc in response["context"]]
        })
    return answers