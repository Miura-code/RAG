import os
from time import strftime
import pandas as pd
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.gpt4all import GPT4All

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logging import get_std_logging

def main():
    logger.info("--> Start Dataset setting.")
    dataset = load_dataset('medical_dialog', 'processed.en', trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])

    dialog = []

    # 患者と医者の発言をそれぞれ抽出した後、順にリストに格納
    patient, doctor = zip(*df['utterances'])
    for i in range(len(patient)):
        dialog.append(patient[i])
        dialog.append(doctor[i])

    df_dialog = pd.DataFrame({"dialog": dialog})

    # 成形終了したデータセットを保存
    df_dialog.to_csv('medical_data.txt', sep=' ', index=False)
    logger.info("--> End Dataset setting. Save dataset to '{}'.".format('medical_data.txt'))
    logger.info("--> Start LLM Model setting.")

    loader = TextLoader('medical_data.txt', encoding="utf-8")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    indexer = VectorstoreIndexCreator(embedding=embeddings)
    index = indexer.from_loaders([loader])
    logger.info("--> Vector Store completed")
    
    # モデルのパラメータファイルがあるローカルの場所を指定。
    # https://huggingface.co/orel12/ggml-gpt4all-j-v1.3-groovy/blob/main/ggml-gpt4all-j-v1.3-groovy.bin
    # llm_name = 'https://huggingface.co/orel12/ggml-gpt4all-j-v1.3-groovy/resolve/main/ggml-gpt4all-j-v1.3-groovy.binn'  # replace with your desired local file path
    # callbacks = [StreamingStdOutCallbackHandler()]
    # llm = GPT4All(model=llm_name, callbacks=callbacks, verbose=True, backend='gptj')
    
    # HaggingfacePipeLineを使う場合
    # llm_name = "meta-llama/Llama-2-japanese"
    llm_name = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = HuggingFacePipeline(
        model=AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", low_cpu_mem_usage=True),
        tokenizer=tokenizer)    
    
    logger.info("--> End LLM Model setting -> {}".format(llm_name))
    
    # クエリと最も類似(k=4)しているドキュメントをindexから検索する
    query = "what is the solution for soar throat"
    results = index.vectorstore.similarity_search(query, k=4)
    context = "\n".join([document.page_content for document in results])
    logger.info("--> Searched context from index. \n{}".format(context))

    template = """
    Please use the following context to answer questions.
    Context: {context}
    ---
    Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, 
                            input_variables=["context", "question"]).partial(context=context)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    logger.info(llm_chain.run("what is the solution for soar throat"))

if __name__ == "__main__":
    
    path = "./results/"
    name = '{}'.format(strftime("%Y%m%d-%H%M%S"))
    logger = get_std_logging(os.path.join(path, "{}.log".format(name)))
    main()