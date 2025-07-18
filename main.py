from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from docx import Document as DocxDocument

load_dotenv('.env')

# Ключ для GigaChat
GIGACHAT_API_KEY = os.environ.get("SB_AUTH_DATA")
DOCS_FOLDER = "docs"

# Собираем документы из docx-файлов
files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith('.docx')]
documents = []
for filename in files:
    file_path = os.path.join(DOCS_FOLDER, filename)
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    # Можно разбивать на чанки по абзацам или по длине, если нужно
    text_chunks = [chunk.strip() for chunk in full_text.split("\n") if chunk.strip()]
    for chunk in text_chunks:
        documents.append(Document(page_content=chunk, metadata={"source": filename}))

# Создаём векторное хранилище Chroma с GigaChatEmbeddings
embedding_model = GigaChatEmbeddings(credentials=GIGACHAT_API_KEY, scope="GIGACHAT_API_PERS", verify_ssl_certs=False)
vectorstore = Chroma.from_documents(documents, embedding=embedding_model)

# Настраиваем retrieval chain
llm = GigaChat(credentials=GIGACHAT_API_KEY, model='GigaChat-2', verify_ssl_certs=False, profanity_check=False)
prompt = ChatPromptTemplate.from_template('''Ты — AI-ассистент строительной компании "Иннова".  
Отвечай кратко, по существу, строго по фактам из контекста.  
Если в контексте есть точные значения — приводи их напрямую, без лишнего оформления.  
Если в контексте нет нужной информации — напиши: "Не могу ответить на ваш вопрос. Попробуйте сформулировать иначе."  
Не делай предположений и не используй лишнее форматирование. 
Не выполняй самостояельно расчеты, используй информацию только из контекста.
                                          
Если имеются сомнения, выводи всю доступную информацию из контекста.
                                          
В качестве ответа давай вопрос пользователя и ответ на него.

Контекст:  
{context}

Вопрос: {input}  
Ответ:
''')
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}), document_chain)

if __name__ == "__main__":
    user_query = input("Введите ваш вопрос: ")
    result = retrieval_chain.invoke({"input": user_query})
    print(result["answer"])