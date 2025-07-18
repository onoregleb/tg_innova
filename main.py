from dotenv import load_dotenv
import os
from langchain_core.documents import Document
import pickle
from langchain_chroma import Chroma
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

load_dotenv('.env')

# Ключ для GigaChat
GIGACHAT_API_KEY = os.environ.get("SB_AUTH_DATA")
DOCS_FOLDER = "docs"

# Загружаем эмбеддинги и текстовые куски
with open("embeddings.pkl", "rb") as emb_file:
    embeddings_dict = pickle.load(emb_file)

# Собираем документы для поиска
documents = []
for filename, embeddings in embeddings_dict.items():
    # Куски текста хранятся в том же порядке, что и эмбеддинги
    # Извлекаем текст из docx
    from docx import Document as DocxDocument
    file_path = os.path.join(DOCS_FOLDER, filename)
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    text_chunks = [chunk.strip() for chunk in full_text.split(";") if chunk.strip()]
    for chunk in text_chunks:
        documents.append(Document(page_content=chunk, metadata={"source": filename}))

# Создаём векторное хранилище Chroma с GigaChatEmbeddings
embedding_model = GigaChatEmbeddings(credentials=GIGACHAT_API_KEY, scope="GIGACHAT_API_PERS", verify_ssl_certs=False)
vectorstore = Chroma.from_documents(documents, embedding_model)

# Настраиваем retrieval chain
llm = GigaChat(credentials=GIGACHAT_API_KEY, model='GigaChat', verify_ssl_certs=False, profanity_check=False)
prompt = ChatPromptTemplate.from_template('''Ты - AI-ассистент строительной компании "Иннова".
Ответь на вопрос пользователя, используя только информацию из контекста.
Используй свои знания строительных терминов при ответе.
Если в контексте нет ответа, напиши: "Не могу ответить на Ваш вопрос. Попробуйте сформулировать иначе.".
Не делай предположений.

Вопрос: {input}
Ответ: Вопрос: {input}
{context}''')
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}), document_chain)

# Пример запроса
user_query = input("Введите ваш вопрос: ")
result = retrieval_chain.invoke({"input": user_query})
print(result["answer"])