# https://huggingface.co/medicalai/ClinicalBERT
# This model card describes the ClinicalBERT model, which was trained on a large multicenter dataset with a large corpus of 1.2B words
#   of diverse diseases we constructed.
#   We then utilized a large-scale corpus of EHRs from over 3 million patient records to fine tune the base language model.
# We used a batch size of 32, a maximum sequence length of 256, and a learning rate of 5e-5 for pre-training our models.
# The ClinicalBERT model was trained on a large multicenter dataset with a large corpus of 1.2B words of diverse diseases we constructed.

import json
import os

import pandas as pd
import torch
from datasets import load_metric
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoConfig, DistilBertForSequenceClassification, DistilBertTokenizer, \
    get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from utils_clinic import df_token_counts, create_data_loader, organize_data, clean_directory


# Prepare the dataset
class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to predict the disease
def predict_disease(symptoms, model, tokenizer, disease_mapping, MAX_LEN):

    inputs = tokenizer(symptoms, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    label_mapping = {v: k for k, v in disease_mapping.items()}

    return label_mapping[predictions.item()]

def finetune_model(trainer, saved_ft_model_dir):

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(saved_ft_model_dir)
    trainer.tokenizer.save_pretrained(saved_ft_model_dir)

def compute_metrics(pred):
    metric_acc = load_metric("accuracy")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
    metric_f1 = load_metric("f1")

    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    acc = metric_acc.compute(predictions=predictions, references=labels)
    precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": acc["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }
def set_trainer(model, training_args, train_data_loader, test_data_loader, tokenizer, compute_metrics,
                learning_rate, use_scheduler=False):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_loader.dataset,
        eval_dataset=test_data_loader.dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if use_scheduler:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_data_loader) * num_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        trainer.optimizers = (optimizer, scheduler)
    return trainer

if __name__ == "__main__":

    # Define the base directory
    base_dir = os.path.dirname(__file__)

    #==============================================
    # model_name          = 'medicalai/ClinicalBERT'
    model_name          = 'emilyalsentzer/Bio_ClinicalBERT'

    bertTypeClassName   = BertForSequenceClassification
    # bertTypeClassName       = DistilBertForSequenceClassification

    bertTokenizerClass  = BertTokenizer
    # bertTokenizerClass  = DistilBertTokenizer

    # saved_ft_model_dir  = 'C:/fine_tune_models/disease_classification'
    saved_ft_model_dir  = 'C:/fine_tune_models/disease_classification/checkpoint-5416'
    logs_dir            = 'C:/tmp/logs'

    train_data_file     = base_dir + '/diseases_symptoms/symptom-disease-train-dataset.csv'
    test_data_file      = base_dir + '/diseases_symptoms/symptom-disease-test-dataset.csv'
    mapping_file        = base_dir + '/diseases_symptoms/mapping.json'

    # train_data_file     = base_dir + '/Medical-Abstracts-TC-Corpus/medical_tc_train.csv'
    # test_data_file      = base_dir + '/Medical-Abstracts-TC-Corpus/medical_tc_test.csv'
    # mapping_file        = base_dir + '/Medical-Abstracts-TC-Corpus/mapping.json'

    BATCH_SIZE          = 8
    MAX_LEN             = 420 #200
    learning_rate       = 1.e-5
    num_epoch           = 35
    test_size           = 0.1

    start_fine_tune     = True
    resume_fine_tune    = False
    do_eval             = True

    use_scheduler       = False
    #==============================================

    config = AutoConfig.from_pretrained(model_name)
    print('Configuration of ' + model_name + ':\n', config)
    seq_len = config.max_position_embeddings

    with open(mapping_file) as f:
        disease_mapping_orig = json.load(f)

    train_data1 = pd.read_csv(train_data_file)
    test_data1 = pd.read_csv(test_data_file)

    train_data, test_data, disease_mapping = organize_data(train_data1, test_data1, disease_mapping_orig, test_size)

    print('\ntrain_data shape:', train_data.shape)
    print('\ntest_data shape:', test_data.shape)

    df_token_counts(train_data, MAX_LEN)
    df_token_counts(test_data, MAX_LEN)

    # Initialize the tokenizer
    tokenizer = bertTokenizerClass.from_pretrained(model_name)
    VOCAB_SIZE = tokenizer.vocab_size
    print('\nVOCAB_SIZE:', VOCAB_SIZE, '\n')
    NUM_CLASSES = len(disease_mapping)
    print('\nNUM_CLASSES:', NUM_CLASSES, '\n')

    # Initialize the Model
    model = bertTypeClassName.from_pretrained(model_name, num_labels=NUM_CLASSES)

    train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

    if start_fine_tune or resume_fine_tune:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=saved_ft_model_dir,
            num_train_epochs=num_epoch,  # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=logs_dir,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",  # Save the model at every epoch
            save_total_limit=2,  # Limit the total number of saved checkpoints
            learning_rate=learning_rate  # 1e-5
        )
    if start_fine_tune:
        # Clean the directories
        clean_directory(logs_dir)
        clean_directory(saved_ft_model_dir)

        trainer = set_trainer(model, training_args, train_data_loader, test_data_loader, tokenizer, compute_metrics,
                              learning_rate, use_scheduler=use_scheduler)

        print("Environment cleaned. Ready for re-training.")
        finetune_model(trainer, saved_ft_model_dir)

    # Resume from the last checkpoint:
    if resume_fine_tune:
        last_checkpoint = None
        if os.path.isdir(saved_ft_model_dir):
            checkpoints = [os.path.join(saved_ft_model_dir, d) for d in os.listdir(saved_ft_model_dir) if d.lower().startswith("checkpoint")]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getctime)

        print('\nFine-Tuning will resume from checkpoint:', last_checkpoint, '\n')
        # load model and tokenizer
        model = bertTypeClassName.from_pretrained(last_checkpoint)
        tokenizer = bertTokenizerClass.from_pretrained(last_checkpoint)

        trainer = set_trainer(model, training_args, train_data_loader, test_data_loader, tokenizer, compute_metrics,
                              learning_rate, use_scheduler=use_scheduler)
        trainer.train(resume_from_checkpoint=last_checkpoint)

    if do_eval:

        trainer = set_trainer(model, training_args, train_data_loader, test_data_loader, tokenizer, compute_metrics,
                              learning_rate, use_scheduler=use_scheduler)
        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")

    # Test
    model = bertTypeClassName.from_pretrained(saved_ft_model_dir)
    tokenizer = bertTokenizerClass.from_pretrained(saved_ft_model_dir)

    '''
    From test set
    Migrain
    Peptic Ulcer Disease
    Urinary Tract Infection
    Gastroenteritis
    Alcoholic Hepatitis
    '''
    symptoms_list = [
        "acidity,indigestion,headache,blurred_and_distorted_vision,excessive_hunger,stiff_neck,depression,irritability,visual_disturbances",
        "I have been having bloody stools for a while now, and I am also feeling weak. I think I may have anaemia.",
        "I've been feeling really down lately, and my pee smells like rotten eggs. My kidney area hurts, and I can't seem to hold my urine in. I have to go to the bathroom all the time.",
        "vomiting,sunken_eyes,dehydration,diarrhoea",
        "vomiting,yellowish_skin,abdominal_pain,swelling_of_stomach,distention_of_abdomen,history_of_alcohol_consumption,fluid_overload"
    ]
    for symptom in symptoms_list:
        predicted_disease = predict_disease(symptom, model, tokenizer, disease_mapping, MAX_LEN)
        print(f'''
        {symptom} 
        {predicted_disease}
        -------------------------------------
        '''
        )
