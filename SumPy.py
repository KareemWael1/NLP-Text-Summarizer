import tkinter as tk
from tkinter import scrolledtext
from transformers import pipeline
import warnings
import os
import logging

# Disable unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Loading BART Summarizer...")
bart = pipeline("text2text-generation", model="ahmeddsakrr/text_summarizer_bart", tokenizer="facebook/bart-base", temperature=1)
print("Loading Pegasus Summarizer...")
pegasus = pipeline("text2text-generation", model="ahmeddsakrr/text_summarizer_pegasus", tokenizer="google/pegasus-xsum", temperature=1)
print("Loading T5 Summarizer...")
t5 = pipeline("text2text-generation", model="ahmeddsakrr/text_summarizer_t5", tokenizer="t5-small", temperature=1)
print("Summarizers Loaded!")
print("Using Bart Summarizer")
summarizer = bart


def summarize_text():
	input_text = input_textbox.get("1.0", tk.END).strip()
	max_length_input = max_len.get()
	if max_length_input.isdigit():
		max_length = int(max_length_input)
	else:
		max_length = 50  # Default max length
	if input_text:
		summarized_text = summarizer(input_text, max_length=max_length)[0]['generated_text']
		output_textbox.config(state=tk.NORMAL)
		output_textbox.delete("1.0", tk.END)
		output_textbox.insert(tk.END, summarized_text)
		output_textbox.config(state=tk.DISABLED)


def validate_max_len(P):
	if P.isdigit() or P == "":
		return True
	return False


def on_model_change(*args):
	load_summarizer(model_var.get())


def load_summarizer(model_name):
	global summarizer
	if model_name == "BART":
		summarizer = bart
		print("Using Bart Summarizer")
	elif model_name == "Pegasus":
		summarizer = pegasus
		print("Using Pegasus Summarizer")
	elif model_name == "T5":
		summarizer = t5
		print("Using T5 Summarizer")


root = tk.Tk()
root.title("SumApp")

# Input
input_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
input_textbox.pack(padx=10, pady=10)

# Model
model_var = tk.StringVar(value="BART")
model_label = tk.Label(root, text="Choose Model:")
model_label.pack(pady=5)
model_dropdown = tk.OptionMenu(root, model_var, "BART", "Pegasus", "T5")
model_dropdown.pack(pady=5)
model_var.trace_add('write', on_model_change)

# Max Length
max_len_label = tk.Label(root, text="Summary Max Length:")
max_len_label.pack(pady=5)
vcmd = (root.register(validate_max_len), '%P')
max_len = tk.Entry(root, validate="key", validatecommand=vcmd, width=10)
max_len.pack(pady=5)

# Summarize Button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text)
summarize_button.pack(pady=10)

# Output
output_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
output_textbox.pack(padx=10, pady=10)
output_textbox.config(state=tk.DISABLED)

root.mainloop()
