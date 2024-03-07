import os
from google.cloud import aiplatform
import vertexai

PROJECT_ID = "mygenaiproject-416315"  
LOCATION = "asia-southeast1" 

vertexai.init(project=PROJECT_ID, location=LOCATION)

def create_session():
    chat_model = vertexai.language_models.ChatModel.from_pretrained("chat-bison@002")
    chat = chat_model.start_chat()
    return chat

def response(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    result = chat.send_message(message, **parameters)
    return result.text

def s(*a):
    print(len(a))


def run_chat():
    s(12,24,232)
    
    chat_model = create_session()
    print(f"Chat Session created")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        content = response(chat_model, user_input)
        print(f"AI: {content}")

if __name__ == '__main__':
    run_chat()
