import os
import openai
import gradio as gr

openai.api_key = os.getenv("OPENAI_API_KEY")

def chatbot(input_text):
    response = openai.Completion.create(
        model="davinci:ft-personal-2023-04-03-04-12-25",
        prompt=f"{input_text}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["END"]
    )
    return response.choices[0].text.strip()

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="输入您关于航班的问题"),
                     outputs="text",
                     title="定制训练的AI航班问答机器人")

iface.launch(share=True)
