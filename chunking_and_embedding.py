from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


loader = CSVLoader(file_path="datasets/data.csv")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_documents(docs, embeddings)

vectorstore.save_local("faiss_index")