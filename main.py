import gradio as gr
from chatbot_logic import answer_from_sources

CUSTOM_CSS = """
.gradio-container { max-width: 1100px; margin: 20px auto; }
.chat-container { background: linear-gradient(180deg,#0f172a, #071028); border-radius: 12px; padding: 18px; color: #e6eef8; }
.header-row { display:flex; gap:12px; align-items:center; margin-bottom:12px; }
.header-row h1 { margin:0; font-size:1.3rem; color: #ff8a00; }
#user_input textarea { background: #0b1220; color: #e6eef8; border-radius:8px; }
#url textarea, #pdf_file input { background:#071027; color:#e6eef8; border-radius:8px; }
.gr-button { border-radius: 10px; padding: 8px 16px; }
.gradio-container .chatbot > div { border-radius: 8px; }
"""

def build_system_notice():
    return (
        "Generative AI Suite — Upload PDF, paste URL, or just ask a question.\n"
        "Toggle Voice to play answer. Clear Chat resets conversation."
    )

def chat_endpoint(user_input, pdf_file, url, use_tts, state):
    pdf_path = pdf_file if pdf_file else None
    url_input = url.strip() if url else None
    user_input = (user_input or "").strip()
    history = state.get("history", [])

    if not any([user_input, pdf_path, url_input]):
        history.append({"role": "system", "content": "Please enter a question, upload a PDF, or provide a URL."})
        state["history"] = history
        return history, state, None, ""

    answer, audio = answer_from_sources(user_input=user_input, pdf_path=pdf_path, url=url_input, state=state, use_tts=use_tts)

    if user_input:
        history.append({"role": "user", "content": user_input})
    if url_input and not user_input:
        history.append({"role": "user", "content": f"URL provided: {url_input}"})
    history.append({"role": "assistant", "content": answer})
    state["history"] = history

    return history, state, audio, ""

def clear_history():
    return [], {"history": []}, None, ""

with gr.Blocks(css=CUSTOM_CSS, title="Generative AI Suite — Chatbot") as demo:
    gr.HTML("<div class='chat-container'><div class='header-row'><h1>Generative AI Suite</h1>"
            "<div style='color:#b7c7ff'>PDF + Website + Voice (gTTS)</div></div>"
            f"<div style='font-size:0.9rem; color:#dbeafe'>{build_system_notice()}</div></div>")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", height=550, type="messages")
            with gr.Row():
                submit = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear Chat")
                use_tts = gr.Checkbox(label="Play voice answer (gTTS)", value=True)
        with gr.Column(scale=2):
            user_input = gr.Textbox(lines=1, placeholder="Type your question here...")
            url = gr.Textbox(lines=1, placeholder="Paste website URL (https://...)", label="Website URL")
            pdf_file = gr.File(file_types=[".pdf"], file_count="single", label="Upload PDF (optional)", type="filepath")
            output_audio = gr.Audio(label="Voice (gTTS)", type="filepath")

    state = gr.State({"history": []})

    submit.click(chat_endpoint, inputs=[user_input, pdf_file, url, use_tts, state],
                 outputs=[chatbot, state, output_audio, user_input])
    user_input.submit(chat_endpoint, inputs=[user_input, pdf_file, url, use_tts, state],
                      outputs=[chatbot, state, output_audio, user_input])
    clear.click(clear_history, inputs=[], outputs=[chatbot, state, output_audio, user_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
