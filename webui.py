import sys,time
from datetime import datetime
import midi2audio
import transformers
from transformers import AutoModelForCausalLM
import requests
from IPython.display import Audio
import os
from anticipation import ops
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *
from modules import ui, shared
import glob
from os.path import dirname, abspath

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
def my_get(url, **kwargs):
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)
original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get


current_directory = dirname(abspath(__file__))
models_dir=current_directory+"/models/*/"
def get_choices(models_dir):
    choices = []
    for i in glob.glob(models_dir):
        split_dir=i.split('/')
        choices.append(split_dir[len(split_dir)-2])
    return choices



last_model_name = ""
model = None
def synthesize(model_name : str,length : int):
    global model
    global last_model_name
    if(model_name=="" or model_name!=last_model_name):
        model = AutoModelForCausalLM.from_pretrained(current_directory+"/models/"+model_name).cuda()
        last_model_name=model_name
    unconditional_tokens = generate(model, start_time=0, end_time=length, top_p=.98)
    mid = events_to_midi(unconditional_tokens)
    now = datetime.now()
    path = current_directory+"/results/" + now.strftime("%Y%m%d_%H%M%S") + ".mid"
    mid.save(path)
    return path



with gr.Blocks(analytics_enabled=False) as webui:
    with gr.Tab("Inference"):
        with gr.Row():
            gr.Markdown("""Anticipation WebUI""")
            model_name = gr.Dropdown(choices=get_choices(models_dir), value='None',
                                       label='Model', info='The model to use for inference.', interactive=True)
            ui.create_refresh_button(model_name, lambda: None,
                                     lambda: {'choices':get_choices(models_dir)},
                                     'refresh-button')
            length = gr.Number(label='Length', value=10, interactive=True)

        submit = gr.Button("Submit")
        output = gr.Audio(label="Generated Music", type="filepath")

        submit.click(synthesize, inputs=[model_name,length], outputs=[output])
if shared.args.listen:
    webui.launch(share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
else:
    webui.launch(share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
