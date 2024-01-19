import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chromadb
import gradio as gr

checkpoint = "LaMini-T5-738M"
persist_directory = "db"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto", 
    torch_dtype=torch.float32
)
# @st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
# @st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    reteriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = reteriever,
        return_source_documents = True
    )
    return qa


def process_answer(instruction, history):
    print(instruction)
    response = ''
    instruction = instruction
    qa = qa_llm()
    page = ''
    generated_text = qa(instruction)
    answer = generated_text['result']
    i = 1
    # print(generated_text)
    if 'source_documents' in generated_text:
        for docs in generated_text['source_documents']:
            page_content,metadata, type = docs
            print(page_content[1])
            print(metadata[1]['source'])
            print(type)
            page = page +  "\n"+  f"({i}) "+ page_content[1] + "\n" + metadata[1]['source'] + "\n"
            i = i+1    

    answer = answer + "\n\n" + page
    print(answer)
    return answer

def main():
    question = "What is lift"
    answer = process_answer(question, None)
    print(answer)
    # st.title("Search for FAA Questions")
    # with st.expander("About the app"):
    #     st.markdown(
    #         """
    #             This app is designed to help you find questions about the FAA.
    #         """
    #     )
    # question = st.text_area("Enter your question here:")
    # if st.button("Search"):
    #     st.info("Your question is: " + question)
    #     st.info("Your answer is")
    #     answer, metadata = process_answer(question)
    #     st.write(answer)
    #     st.write(metadata)
    # gr.ChatInterface(
    #     process_answer,
    #     chatbot=gr.Chatbot(height=500),
    #     textbox=gr.Textbox(placeholder="Enter your question here",container=False, scale=7),
    #     title="Search for FAA Questions",
    #     theme="soft",
    #     examples=[
    #         "What is lift",
    #         "What is drag",
    #         "What is Private Pilot Written Test"
    #     ],
    #     cache_examples=False,
    #     retry_btn=None,
    #     undo_btn="Delete Previous",
    #     clear_btn="Clear"
    # ).launch(share=True)

    chat = gr.ChatInterface(
        process_answer,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Enter your question here",container=False, scale=7),
        title="Search for FAA Questions",
        theme="soft",
        examples=[
            "What is lift",
            "What is drag",
            "What is Private Pilot Written Test"
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear"
    )
chat = gr.ChatInterface(
        process_answer,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Enter your question here",container=False, scale=7),
        title="Search for FAA Questions",
        theme="soft",
        examples=[
            "What is lift",
            "What is drag",
            "What is Private Pilot Written Test"
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear"
    )
if __name__ == "__main__":
    main()