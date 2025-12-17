import os
from flask import Flask, request, jsonify, render_template, send_from_directory 
from flask_cors import CORS
from langchain_groq import ChatGroq

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = None

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

@app.route('/')
def serve_main_page():
    return render_template('index.html') 

@app.route('/chat_ui')
def serve_chat_ui():
    return render_template('chat.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/chat', methods=['POST'])
def chat():
    if not llm:
        return jsonify({"respuesta": "Servidor conectándose..."})
    data = request.json
    mensaje_usuario = data.get('mensaje')
    try:
        prompt = f"Eres UniBot, el asistente de la UNCAUS. Responde breve: {mensaje_usuario}"
        respuesta = llm.invoke(prompt)
        return jsonify({"respuesta": respuesta.content})
    except Exception:
        return jsonify({"respuesta": "Lo siento, intenta de nuevo en un momento."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))
