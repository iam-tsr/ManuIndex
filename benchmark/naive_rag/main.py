from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class NaiveRAG:
    def __init__(
            self, 
            embeddings,
            chunk_size=500, 
            top_k=3,
    ):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embeddings = embeddings

    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, documents):
        vector_store = FAISS.from_texts(documents, embedding=self.embeddings)
        vector_store.save_local("faiss_index")

    def get_relevant_context(self, user_question):
        new_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=self.top_k)
        return docs
    
    def main(self, documents, user_question):
        text_chunks = self.get_chunks(documents)
        self.get_vector_store(text_chunks)
        docs = self.get_relevant_context(user_question)
        context = " ".join([doc.page_content for doc in docs])
        return context