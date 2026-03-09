import os

# Step 0: Load environment variables from .env when available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'
HF_TOKEN = os.getenv('HF_TOKEN')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_local_llm_path(model_id: str) -> str:
    # Map a model id like owner/model into a safe local folder name.
    safe_model_name = model_id.replace('/', '__')
    return os.path.join(BASE_DIR, 'local_models', safe_model_name)

custom_prompt_template = """
Answer the question based only on the following context:
{context}
You are allowed to rephrase the answer based on the context.
Keep the response concise (2-4 sentences).
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Step 1: Create the prompt template that controls answer style and length.
def set_custom_prompt():
    """
    Build the template used by the retriever + LLM chain.
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    # Retrieve top matching chunks from FAISS, then generate an answer from them.
    # k=2 means the retriever returns the 2 most relevant chunks per question.
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Step 2: Load or download the language model used to generate answers.
    # Prefer a previously downloaded local model to avoid repeated downloads.
    model_id = os.getenv('LLM_MODEL_ID', 'HuggingFaceTB/SmolLM2-360M-Instruct')
    local_llm_path = os.getenv('LOCAL_LLM_PATH', _default_local_llm_path(model_id))

    local_model_ready = os.path.isdir(local_llm_path) and os.path.exists(
        os.path.join(local_llm_path, 'config.json')
    )

    if local_model_ready:
        # Fast path: load model files from disk only.
        tokenizer = AutoTokenizer.from_pretrained(local_llm_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_llm_path, local_files_only=True)
    else:
        # Fallback path: download once, then cache under local_models/ for next runs.
        tokenizer_kwargs = {}
        model_kwargs = {}
        if HF_TOKEN:
            tokenizer_kwargs['token'] = HF_TOKEN
            model_kwargs['token'] = HF_TOKEN

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Save downloaded model files so future runs are offline and faster.
        os.makedirs(local_llm_path, exist_ok=True)
        tokenizer.save_pretrained(local_llm_path)
        model.save_pretrained(local_llm_path)

    # Wrap model+tokenizer into a text-generation pipeline LangChain can call.
    text_generation_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

# QA Model Function
def qa_bot():
    # Step 3: Build the full QA pipeline (embeddings + retriever + LLM + prompt).
    # Initialize embeddings
    embedding_kwargs = {'device': 'cpu'}
    if HF_TOKEN:
        embedding_kwargs['token'] = HF_TOKEN

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs=embedding_kwargs)
    # Load the existing vector index built by ingest.py.
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load the LLM model
    llm = load_llm()

    # Set the custom prompt
    qa_prompt = set_custom_prompt()

    # Create the QA chain
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    # Step 4: Ask one question and return the raw QA output dictionary.
    # Helper for non-Chainlit usage (e.g., direct function calls/tests).
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response



#chainlit code
if hasattr(cl, 'on_chat_start') and hasattr(cl, 'on_message'):
    @cl.on_chat_start
    async def start():
        # Step 5: On app start, prepare one chain and store it in session memory.
        # Build the QA chain once at chat startup and keep it in session state.
        chain = qa_bot()
        msg = cl.Message(content="Starting the chatbot...")
        await msg.send()
        msg.content = "Q/A About Alzheimer's Disease: Ask me anything about Alzheimer's disease, and I'll provide you with concise answers based on the latest research and information available."
        await msg.update()

        cl.user_session.set("chain", chain)

    @cl.on_message
    async def main(message: cl.Message):
        # Step 6: For each user message, run retrieval + generation and reply.
        # Reuse existing chain per user session; initialize if missing.
        chain = cl.user_session.get("chain")
        if chain is None:
            chain = qa_bot()
            cl.user_session.set("chain", chain)

        res = await cl.make_async(chain.invoke)({"query": message.content})
        # LangChain returns the generated answer and the source documents used.
        answer = res["result"]
        sources = res["source_documents"]
        
        if sources:
            # Show short source citations so answers are traceable.
            citations = []
            for doc in sources[:3]:
                metadata = getattr(doc, "metadata", {}) or {}
                source_name = os.path.basename(metadata.get("source", "unknown"))
                page = metadata.get("page")
                if page is not None:
                    citations.append(f"{source_name} (p.{page})")
                else:
                    citations.append(source_name)

            unique_citations = list(dict.fromkeys(citations))
            # Remove duplicate source names before showing them to the user.
            answer += "\nSources: " + ", ".join(unique_citations)
        else:
            answer += "\nNo sources found"

        await cl.Message(content=answer).send()

    