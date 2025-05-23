import os
import torch
import gradio as gr
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

# Prevent potential Torch Hub issues
torch.classes.__path__ = []

# Load tokenizer and model with LoRA adapter
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    special_tokens = {"additional_special_tokens": ["<AST>", "</AST>"]}
    tokenizer.add_special_tokens(special_tokens)

    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model.resize_token_embeddings(32102)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o", "decoder.q", "decoder.k", "decoder.v", "decoder.o", "wi", "wo"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    checkpoint_path = os.path.join("checkpoints", "mngast120k2", "checkpoint-25000")
    model.load_adapter(checkpoint_path, adapter_name="default")

    return tokenizer, model

# Prediction function
def generate_method_name(code_input):
    if not code_input.strip():
        return "Please paste a valid code snippet."
    
    try:
        tokenizer, model = load_model()
        input_ids = tokenizer(code_input, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids=input_ids, max_length=32)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=generate_method_name,
    inputs=gr.Textbox(lines=50, label="Paste Python/Java Method Code Here"),
    outputs=gr.Textbox(label="Predicted Method Name"),
    title="CodeT5 Method Name Generator",
    description="Enter a Python or Java method. This app uses a fine-tuned CodeT5 model with AST enhancements to predict an appropriate function name."
)

if __name__ == "__main__":
    iface.launch()
