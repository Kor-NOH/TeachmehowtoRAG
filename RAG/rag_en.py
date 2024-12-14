import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangChainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
import numpy as np

# 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "test"
os.environ["OPENAI_API_KEY"] = "test"

# CSV 파일 로드 및 문서로 변환
csv_filepath = "Rag_test_data.csv"
df = pd.read_csv(csv_filepath)

# 헤더를 기반으로 모든 정보를 포함하는 문서 생성
documents = [
    LangChainDocument(
        page_content=", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])]),
        metadata={"source": i}
    )
    for i, row in df.iterrows()
]

# 문서 벡터화 및 FAISS 벡터 저장소 생성
embedding_function = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_store = FAISS.from_documents(documents, embedding_function)

# BM25 초기화
corpus = [doc.page_content for doc in documents]
tokenized_corpus = [doc.split() for doc in corpus]  # 문서 내용을 토큰화
bm25 = BM25Okapi(tokenized_corpus)

# LLM 초기화
lim = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# PromptTemplate 정의
prompt_template = """주어진 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트에 관련 정보가 없다면, “제공된 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

컨텍스트:
{context}

질문: {question}
답변:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# RetrievalQA 파이프라인 초기화
rag_pipeline = RetrievalQA.from_chain_type(
    llm=lim,  # llm 인자에 lim을 전달
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    chain_type_kwargs={"prompt": PROMPT}  # PromptTemplate을 직접 전달
)

# 하이브리드 검색 함수
def hybrid_search(query, k=15):
    # 의미론적 검색
    semantic_docs = vector_store.similarity_search(query, k=k)
    
    # 키워드 기반 검색
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[-k:]
    keyword_docs = [documents[i] for i in reversed(top_n)]

    # 결과 결합 (중복 제거)
    combined_docs = []
    seen = set()
    for doc in semantic_docs + keyword_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc)

    return combined_docs[:k]

# LLM을 사용한 질문 분리 및 처리
def get_combined_answer(query):
    # 질문 분리
    prompt_for_splitting = f"질문: '{query}'\n위의 질문을 개별적인 질문으로 나눠주세요."
    split_response = lim.invoke(prompt_for_splitting)
    sub_queries = [q.strip() for q in split_response.content.split('\n') if q.strip()]

    answers = []

    for sub_query in sub_queries:
        docs = hybrid_search(sub_query, k=15)
        # 중복 제거된 문서 내용 확인
        summarized_content = "\n".join(set(doc.page_content for doc in docs))

        # 컨텍스트와 질문 매칭
        if summarized_content.strip():
            result = rag_pipeline.invoke({"query": sub_query, "context": summarized_content})
            answers.append(f"질문: {sub_query}\n답변: {result['result']}")
        else:
            answers.append(f"질문: {sub_query}\n답변: 제공된 정보로는 답변할 수 없습니다.")

    # 최종 응답 결합
    final_answer = "\n\n".join(answers)
    return final_answer


# 사용 예시
query = "오지우의 전화번호를 알려줘"
answer = get_combined_answer(query)
print("RAG 응답 생성 결과:")
print(answer)