import os
import streamlit as st
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

torch.classes.__path__ = []
from time import sleep

@st.cache_resource
def load_model():
    # Load tokenizer and add special tokens
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    special_tokens = {"additional_special_tokens": ["<AST>", "</AST>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load and configure model
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model.resize_token_embeddings(32102)
    
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q", "k", "v", "o", 
            "decoder.q", "decoder.k", "decoder.v", "decoder.o",
            "wi", "wo",
        ],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Load trained adapter
    base_dir = os.getcwd()
    checkpoint_dir = os.path.join(
        base_dir, "checkpoints", "mngast120k2", "checkpoint-25000"
    )
    model.load_adapter(checkpoint_dir, adapter_name="default")
    
    return tokenizer, model

def generate_output(input_text, tokenizer, model):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids=input_ids, max_length=32)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)



# Streamlit UI
st.title("CodeT5 Function Name Extractor")
st.write("This app predicts function names from code snippets using CodeT5.")

# Load model and tokenizer
tokenizer, model = load_model()

# User input box
user_input = st.text_area(
    "Enter your Python/Java function code:",
    height=300,
    placeholder="Paste function code here..."
)

# Analyze button
if st.button("Analyze Code"):
    if user_input.strip() == "":
        st.warning("Please enter some code to analyze")
    else:
        with st.spinner("Generating analysis..."):
            sleep(1)  # fake processing delay
            try:
                output = generate_output(user_input, tokenizer, model)
                st.subheader("Function Name:")
                st.write(output)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")