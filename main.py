from dotenv import load_dotenv
import os
from langchain_core.documents import Document
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import create_retrieval_chain

# Загрузка переменных окружения
load_dotenv('.env')

# Ключ для GigaChat
giga_key = os.environ.get("SB_AUTH_DATA")


def load_docx(file_path: str) -> str:
    """
    Загружает текст из docx файла.

    Args:
        file_path (str): Путь к файлу docx.

    Returns:
        str: Текст из файла.
    """
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)


def load_all_docx_from_folder(folder_path: str):
    """
    Загружает текст из всех docx файлов в указанной папке.

    Args:
        folder_path (str): Путь к папке.

    Returns:
        List[Document]: Список документов.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            document_text = load_docx(file_path)
            docs.append(Document(page_content=document_text))
    return docs


# Путь к папке с файлами .docx
folder_path = "./docx_folder"
docs = load_all_docx_from_folder(folder_path)

# Разделение текстов на части
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

# Инициализация модели для создания эмбеддингов
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# Создание векторного хранилища
vector_store = FAISS.from_documents(split_docs, embedding)
embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Инициализация GigaChat
llm = GigaChat(credentials=giga_key, model='GigaChat', verify_ssl_certs=False, profanity_check=False)

# Шаблон для запроса
prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста. Если в контексте нет \
информации для ответа, сообщи об этом пользователю. \
Контекст: {context}
Вопрос: {input}
Ответ:'''
                                          )

# Создание цепочек
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
