"""
This module run an interface to edit a specific image
"""
import gradio as gr
import torch
from PIL import Image

from invertor.pte import PivotalTuningEdition

# You can change the path to your custom model
model = PivotalTuningEdition.load("./pte.pkl")

def generate(alpha: float) -> Image:
    """
    Generate the image

    :param alpha: Strength of editing
    :type alpha: float
    :return: Edited image
    :rtype: Image
    """
    return model.edit_pivot(torch.Tensor([alpha]).cuda())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            feature = gr.Slider(minimum=-2, maximum=2, value=0, step=0.1, label="Lips")
            btn = gr.Button("Run")

        output = gr.Image(type="pil")
        btn.click(fn=generate, inputs=[feature], outputs=output) # pylint: disable=E1101

demo.launch() # share=True, auth=("username", "JzshWD`=@.}nA&VaQ>^*B;HU-Ttm7LSj8xR_#M2+p$r:?wYdF~")
