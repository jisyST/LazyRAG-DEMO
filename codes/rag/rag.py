import lazyllm
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker


prompt = 'You will play the role of an AI question-answering assistant and complete a conversation task in which you need to provide your answer based on the given context and question. Please note that if the given context cannot answer the question, do not use your prior knowledge but tell the user that the given context cannot answer the question.'
documents = Document(dataset_path="", embed=OnlineEmbeddingModule(), manager=False)



with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="CoarseChunk", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, group_name="CoarseChunk",  similarity="bm25_chinese", topk=3)
    ppl.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"), topk=1, output_format='content', join=True) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))


if __name__ == "__main__":
    lazyllm.WebModule(ppl, port=23456).start().wait()
