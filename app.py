import gradio as gr
import torch

from invertor.pte import PivotalTuningEdition


model = PivotalTuningEdition.load("./pte.pkl")

def generate(feature):
    return model.edit_pivot(torch.Tensor([feature]).cuda())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            feature = gr.Slider(minimum=-2, maximum=2, value=0, step=0.1, label="Lips")
            btn = gr.Button("Run")

        output = gr.Image(type="pil")
        btn.click(fn=generate, inputs=[feature], outputs=output)

demo.launch() # share=True, auth=("username", "JzshWD`=@.}nA&VaQ>^*B;HU-Ttm7LSj8xR_#M2+p$r:?wYdF~")
