import torch
from .distill import load_speech_recognition_model, download_audio_from_youtube
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain_together import ChatTogether
import textwrap
import os

together_api_key = None

def set_together_api_key(api_key):
    global together_api_key
    together_api_key = api_key

def load_transcript(url):
    whisper = load_speech_recognition_model()
    audio_file, name = download_audio_from_youtube(url)
    transcription = whisper(audio_file, chunk_length_s=30, stride_length_s=5, batch_size=8)
    with open(f"{name}.txt", "w") as f:
        f.write(transcription["text"])

    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"File '{audio_file}' deleted successfully.")
    return f"{name}.txt"

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    response_text = wrap_text_preserve_newlines(llm_response['text'])
    sources_list = [source.metadata['source'] for source in llm_response['context']]
    return {"answer": response_text, "sources": sources_list}

def get_prompt(instruction, sys_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt + instruction + E_INST
    return template

def answer_youtube_question(youtube_url, query):
    if together_api_key is None:
        raise ValueError("Together AI API key is not set. Please set it using the 'set_together_api_key' function.")

    # Load transcript
    transcript_path = load_transcript(youtube_url)

    # Load documents
    loader = TextLoader(transcript_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Load embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}, encode_kwargs={'normalize_embeddings': True})

    # Create FAISS database
    db = FAISS.from_documents(texts, embeddings)

    # Load LLM model
    llm = ChatTogether(model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT", max_tokens=2048, together_api_key=together_api_key)

    # Create prompt
    instruction = "Given the context that has been provided. \n {context}, Answer the following question: \n{question}"
    sys_prompt = """You are an expert in YouTube video question and answering.
    You will be given context to answer from. Answer the questions with as much detail as possible and only in paragraphs.
    In case you do not know the answer, you can say 'I don't know' or 'I don't understand'.
    In all other cases provide an answer to the best of your ability."""
    prompt_template = get_prompt(instruction, sys_prompt)

    # Create retrieval chain
    retriever = db.as_retriever()
    template = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    llm_chain = LLMChain(llm=llm, prompt=template)
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    # Process query
    ans = rag_chain.invoke(query)
    return process_llm_response(ans)
