# METHOD_NAME_GENERATION

This repository contains experiments on method name prediction using fine-tuned variants of the CodeT5 model (Salesforce/codet5-base). Models are trained with LoRA (PEFT) on Java functions from the MGAST dataset, with and without AST-based enhancements.

## Repository Structure

METHOD_NAME_GENERATION/ 
├── checkpoints/ 
    │ 
    ├── adithya/ 
        │ 
        │ ├── mngast30k_session1/ │ 
        │ ├── mngast120k1/ │ 
        │ └── mngast120k2/ │ 
    └── zahra/ 
├── mng_env/ # Python environment files 
├── notebooks/ 
    │ ├── mngast30k.ipynb # Training + inference for 30k subset 
    │ └── mngast120k.ipynb # Training + inference for 120k    
├── mngast30k_app.py # Streamlit app for 30k model 
├── mngast120k_app.py # Streamlit app for 120k model 
├── requirements.txt 
└── README.md


## Setup Instructions

1. Install dependencies: pip install -r requirements.txt

2. To run the Streamlit app locally: streamlit run mngast30k_app.py OR streamlit run mngast120k_app.py

Note: Ensure the correct adapter checkpoint directory is set inside the app file when loading with `model.load_adapter(...)`.


## Best Checkpoints to Use

| Model              | Best Checkpoint        | Notes                                                             |
|--------------------|------------------------|-------------------------------------------------------------------|
| mngast30k_session1 | checkpoint-18750       | Best evaluation loss                                              |
| mngast120k1        | checkpoint-25000OR15000| 25000 is latest, but 15000 has slightly lower eval loss           |
| mngast120k2        | checkpoint-25000       | Final and stable version                                          |


### Important Note on Trainer

- When **testing predictions** or **evaluating the model**, you **do not need to initialize or run the Trainer**.
- To **resume training**, you must:
  - Manually load the adapter from the correct checkpoint.
  - Run `trainer.train()`.

This behavior is demonstrated in the code block under the comment "Pls verify file directory" in the notebook.
