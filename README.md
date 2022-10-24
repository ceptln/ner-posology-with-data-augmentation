# NER_Posology_Extraction

This repo performs Named Entity Recognition for Posology Data. The data comes from Anonymized Hospital Admission in France. 

In this repo we explore various data augmentation tasks : 
- back translation
- paraphrase generation
- random deletion / random swapping
- summarization
- synonym generation

We then use the augmented data to fit a transformers model ```CamembertForTokenClassification```[here](https://huggingface.co/camembert-base) from Huggingface's transformers library.

### Create your environment

With Python 3.8.10 (default on Ubuntu 20.04), create your venv with ```make init-venv``` and activate it with ```source .venv/bin/activate```
Please not that you might need to use ```sudo apt install python3.8-venv``` in the first place in order to be able to create venv

### Perform data augmentation

You can perform data augmentation by using ```python src/data-augmentation/augmentation.py```. This will create a ```data_augmented.jsonl``` file and save it in ```data```

### Full pipeline usage

You can active the full pipeline, training and predicting by running ```python main.py```. This will train the model, plot vizualisations and predict using  ```test.csv ``` and generate a submission file. 

