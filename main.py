from fastapi import FastAPI
import gradio as gr

from app import chat

app = FastAPI()

@app.get("/")
async def root():
    return 'Gradio app is running at /gradio', 200


app = gr.mount_gradio_app(app, chat, path='/gradio')
