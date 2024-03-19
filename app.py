from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel
from vertexai.language_models import TextGenerationModel
import os

app = Flask(__name__)


def create_chat_model():
    PROJECT_ID = "mygenaiproject-416315"  
    LOCATION = "asia-southeast1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    chat_model = ChatModel.from_pretrained("chat-bison@002")
    chat = chat_model.start_chat()
    return chat

def responseChat(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    result = chat.send_message(message, **parameters)
    return result.text

def create_tuned_model():
    PROJECT_ID = "768537793492"  
    LOCATION = "us-central1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    chat_model = TextGenerationModel.from_pretrained("text-bison@002")
    chat_model = chat_model.get_tuned_model("projects/768537793492/locations/us-central1/models/5127349275272937472")
    return chat_model

def responseTunedChat(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    
    result = chat.predict(message,**parameters)
    return result.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbotTunedModel')
def chatbotTunedModel():
    return render_template('chatbotTunedModel.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertex_palm():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    chat_model = create_chat_model()
    content = responseChat(chat_model,user_input)
    return jsonify(content=content)

@app.route('/tunedModel', methods=['GET', 'POST'])
def vertex_tunedModel():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    context = "Classify the above text in to one of the matching 'output_text' classes like [Allergy / Immunology, Autopsy, Bariatrics, Cardiovascular / Pulmonary, Chiropractic, Cosmetic / Plastic Surgery]"
    user_input = user_input + context
    chat_model = create_tuned_model()
    content = responseTunedChat(chat_model,user_input)
    return jsonify(content=content)


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
