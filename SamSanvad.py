from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import openai
import json
import pyttsx3
import speech_recognition as sr
from googletrans import Translator



app = Flask(__name__, static_folder='static')

# Text-to-speech engine setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed of voice

translator= Translator()
# Paths
UPLOAD_FOLDER = 'uploads'
INDEX_FOLDER = 'indices'
MAPPING_FILE = 'mappings.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)


# Global Variables for Settings (can later be stored in a file or database)
settings = {"role": "", "context": ""}

@app.route('/')
def home():
    return render_template('samsanvad.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    index_file_name = f"{file.filename}"
    index_path = os.path.join(INDEX_FOLDER, index_file_name)
    vector_store.save_local(index_path)

    mapping = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
    mapping[file.filename] = index_file_name
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f)

    return index_path


@app.route('/indices', methods=['GET'])
def get_indices():
    try:
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, 'r') as f:
                mapping = json.load(f)
            indices_with_files = [{"pdf_file": pdf, "index_file": index} for pdf, index in mapping.items()]
        else:
            indices_with_files = []
    except json.JSONDecodeError:
        return jsonify({"error": "Mapping file is corrupted. Please reset or fix the file."}), 500

    return jsonify(indices_with_files)


@app.route('/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(UPLOAD_FOLDER, filename)
    else:
        return "Error: PDF file not found.", 404


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('question')
    index_name = data.get('index')

    index_path = os.path.join(INDEX_FOLDER, index_name)
    vector_store = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=openai.api_key),
                                    allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    relevant_docs = retriever.get_relevant_documents(user_input)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Incorporate role and context from settings
    role = settings.get("role", "You are a helpful assistant.")
    context += " " + settings.get("context", "")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
        ],
        max_tokens=150,
        temperature=0.7
    )

    bot_reply = response['choices'][0]['message']['content'].strip()

    selected_language = settings.get("language", "english")
    print (selected_language)
    if selected_language != "english":
        try:
            bot_reply = translator.translate(bot_reply, dest=selected_language).text
        except Exception as e:
            print(f"Error translating response: {e}")


    # Text-to-speech for chatbot's reply
    tts_engine.save_to_file(bot_reply, 'response_audio.mp3')
    tts_engine.runAndWait()

    return jsonify({"response": bot_reply})

@app.route('/voice_input', methods=['POST'])
def voice_input():
    recognizer = sr.Recognizer()
    user_input ="How Are You"
    try:
        # Retrieve audio file from the POST request
        audio_file = request.files['audio']
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        # Process the audio file
        source= sr.AudioFile("temp.wav")
        print("Recording...")
        audio_data = recognizer.record(source)
        user_input = recognizer.recognize_google(audio_data)
        print("Transcription complete.")

        # Cleanup
        os.remove("temp.wav")

        # Return transcribed text
        return jsonify({"question": user_input})


    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio."})
    except sr.RequestError as e:
        return jsonify({"error": f"API error: {str(e)}"})
    except Exception as e:
        return jsonify({"error": str(e)})
    except:
        return jsonify({"question": user_input})


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/get_audio', methods=['GET'])
def get_audio():
    return send_from_directory('.', 'response_audio.mp3')


# New Route to Save Settings
@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    role = data.get('role')
    context = data.get('context')
    language = data.get('language')

    if role and context and language:
        settings["role"] = role
        settings["context"] = context
        settings["language"] = language  # Save selected language in settings
        return jsonify({"message": "Settings saved successfully!"}), 200
    return jsonify({"message": "Invalid settings data."}), 400


if __name__ == '__main__':
    app.run(debug=True)
