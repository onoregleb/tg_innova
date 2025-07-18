import os
from gigachat import GigaChat
from dotenv import load_dotenv
import pickle
from docx import Document

load_dotenv('.env')

GIGACHAT_API_KEY = os.environ.get("SB_AUTH_DATA")
DOCS_FOLDER = "docs"

client = GigaChat(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)

uploaded_files = {}
embeddings_dict = {}

from langchain_gigachat.embeddings import GigaChatEmbeddings

embedding_model = GigaChatEmbeddings(
    credentials=GIGACHAT_API_KEY,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False
)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

for filename in os.listdir(DOCS_FOLDER):
    if filename.lower().endswith(".docx"):
        file_path = os.path.join(DOCS_FOLDER, filename)
        with open(file_path, "rb") as f:
            file_obj = client.upload_file(f)
            file_id = getattr(file_obj, 'id_', None)
            uploaded_files[filename] = file_id
            print(f"Загружен файл: {filename}, ID: {file_id}")
        
        # Извлечение текста и разбиение по символу ";"
        full_text = extract_text_from_docx(file_path)
        text_chunks = [chunk.strip() for chunk in full_text.split(";") if chunk.strip()]

        embeddings = []
        for chunk in text_chunks:
            embedding = embedding_model.embed_documents([chunk])[0]
            embeddings.append(embedding)

        embeddings_dict[filename] = embeddings
        print(f"Эмбеддингов для {filename} создано: {len(embeddings)}")

# Сохраняем идентификаторы файлов в файл
with open("uploaded_files.txt", "w", encoding="utf-8") as out:
    for fname, fid in uploaded_files.items():
        out.write(f"{fname}: {fid}\n")

# Сохраняем эмбеддинги в файл
with open("embeddings.pkl", "wb") as emb_file:
    pickle.dump(embeddings_dict, emb_file)

print("Все DOCX-файлы из docs загружены в GigaChat. Идентификаторы сохранены в uploaded_files.txt.")
print("Эмбеддинги сохранены в embeddings.pkl.")
