#pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai faiss-cpu pypdf
from langchain_community.document_loaders import CSVLoader, PyPDFLoader

csv_loader = CSVLoader(file_path='C:/Users/mroop/OneDrive/Desktop/Charming_Data/NYC_OpenData_merged/data/NYC_BuildingEnergyWaterData_LocalLaw84_DataDictionary_edited 1.21.25 - Column Information.csv', encoding='UTF-8')
pdf_loader = PyPDFLoader("https://www.nyc.gov/assets/buildings/pdf/ll33_faqs.pdf")

csv_docs = csv_loader.load()
pdf_docs = pdf_loader.load()

docs = csv_docs + pdf_docs

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


OPENAI_API_KEY="sk-None-EOUOIa7NX36gbj8F0X8PT3BlbkFJbvpvdj1TRaTKtGYQi4eu"

text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

docsearch = FAISS.from_documents(docs, embeddings)
docsearch.save_local("faiss_index")

# db = FAISS.from_documents(texts, embeddings)

# retriever = db.as_retriever()

# from langchain.tools.retriever import create_retriever_tool

# rag_tool = create_retriever_tool(
#     retriever, "search_data_dictionary", "Returns information about NYC Local Law 33/18 and denfitions about words related to Building Energy Ratings.",
# )