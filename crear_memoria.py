import os
# Cargador de PDF
from langchain_community.document_loaders import PyPDFLoader
# AQUÍ EL CAMBIO: El cortador de texto ahora viene de aquí
from langchain_text_splitters import CharacterTextSplitter
# Base de datos Chroma
from langchain_community.vectorstores import Chroma
# Embeddings (Cerebro para entender texto)
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Cargar PDF
print("🔄 Cargando documentos...")
documentos = []
if not os.path.exists("documentos"):
    print("❌ ERROR: No encuentro la carpeta 'documentos'. Créala y pon los PDFs ahí.")
else:
    for archivo in os.listdir("documentos"):
        if archivo.endswith(".pdf"):
            ruta_pdf = os.path.join("documentos", archivo)
            print(f"   - Leyendo: {archivo}")
            try:
                loader = PyPDFLoader(ruta_pdf)
                documentos.extend(loader.load())
            except Exception as e:
                print(f"   ⚠️ No se pudo leer {archivo}: {e}")

    if not documentos:
        print("⚠️ No encontré PDFs válidos. Asegúrate de que estén en la carpeta 'documentos'.")
    else:
        # 2. Cortar textos
        print(f"📄 Procesando {len(documentos)} páginas...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos = text_splitter.split_documents(documentos)

        # 3. Crear Memoria (ChromaDB)
        print("💾 Guardando memoria en 'chroma_db'...")
        # Usamos un modelo ligero y gratuito
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Guardar en disco
        db = Chroma.from_documents(textos, embeddings, persist_directory="chroma_db")
        
        # Forzar guardado (para versiones nuevas de Chroma)
        try:
            db.persist() 
        except:
            pass # En versiones nuevas es automático, así que ignoramos si da error este paso.

        print("✅ ¡LISTO! La carpeta 'chroma_db' ha sido creada correctamente.")