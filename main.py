import numpy as np
import pandas as pd
import yaml
import torch
import seaborn    

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, CamembertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import f1_score
from tqdm import trange
from utils import getter, augmentation, preprocessing, pipeline, vizualisation, submission_helper


with open("config.yaml") as f:
    config = yaml.safe_load(f)

###############################################
################## GET DATA ###################
###############################################
df = getter.get_data(config["jsonl_filepath"])

########################################################
################## DATA AUGMENTATION ###################
########################################################
print("Starting data augmentation...")

# to be updated 
# with augmentation functions
df = augmentation.augment_data(df)
# df.loc[:,"has_label"] = df['label'].apply(lambda x: True if len(x)>0 else False)
# df = df.loc[df['has_label'] == True]
# df
#########################################################
################## DATA PREPROCESSING ###################
#########################################################
print("Starting data preprocessing...")

df = preprocessing.reformat_doccano_output(df)

df = preprocessing.pre_tokenize(df)


# to be updated
# Keeping rows with less than 512 pre-tokens because 512 is the max
df["len_pre-tokens"] = df["pre-tokens"].apply(lambda x: len(x))
df.loc[df["len_pre-tokens"] < 512]["len_pre-tokens"]
print(df)
sentences = [row for row in df["pre-tokens"].values]
labels = [row for row in df["labels"].values]

tag2idx = config["tag_values"]
tag2idx = {v: k for k, v in tag2idx.items()}

# to be updated
# Is it necessary to add "PAD" ?
tag2idx["PAD"] = 8
tag_values = list(tag2idx.values())

MAX_LEN = config["MAX_LEN"]
bs = config["bs"]

# Connect to the GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenized_texts_and_labels = [
    pipeline.tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

## ADD PADDING 

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

### Load the inputs to the GPU
tr_inputs.to(device)
val_inputs.to(device)
tr_tags.to(device)
val_tags.to(device)
val_masks.to(device)

## Convert into Tensor

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


#####################################################
#################### MODEL SETUP ####################
#####################################################
print("Starting model setup...")

model = CamembertForTokenClassification.from_pretrained(
    "camembert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

epochs = 15
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

########################################################
#################### MODEL TRAINING ####################
########################################################
print("Starting model training...")

# Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags), average=None))



# # ========================================
# #               Save model
# # ========================================

torch.save(model, "./models/NER_Bert.pt")
print("Model saved!")


# # ========================================
# #               Vizualisation
# # ========================================

confusion_matrix =  seaborn.heatmap(metrics.confusion_matrix(pred_tags, valid_tags))
confusion_matrix.savefig('confusion_matrix.png', dpi=400)

vizualisation.plot_learning_curve(loss_values, validation_loss_values)

#############################################
########## GENERATE SUBMISSION CSV ##########
#############################################

# =============================================
# ========== Generate submission CSV ==========
# =============================================

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ===============================
# ========== Load data ==========
# ===============================
print("Loading data and model ...")
    
test_csv_filepath = ""
submission_csv_filepath = ""

test_sentences, tokens_id = submission_helper.testcsv_to_sentences(test_csv_filepath, sentences_len=100)

# ======================================
# ========== Tokenize and pad ==========
# ======================================
print("Tokenizing and padding ...")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokenized_sentences, tokenized_sentences_ids = submission_helper.tokenize_and_adjust_ids(tokenizer, test_sentences, tokens_id)

submission_helper.check_missing_tokens(tokenized_sentences_ids, test_csv_length=3557)

MAX_LEN = config["MAX_LEN"]
bs = config["bs"]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                          maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")

attention_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]

# ===============================
# ========== To tensor ==========
# ===============================
print("Converting to tensors ...")

valid_dataloader = submission_helper.lists_to_tensors(input_ids, attention_mask)

# ======================================
# ========== Make predictions ==========
# ======================================
print("Making the predictions ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

predictions = submission_helper.make_predictions(device, valid_dataloader, model)

# ========================================
# ========== Predictions to CSV ==========
# ========================================
print("Saving predictions as csv ...")

new_tokens, new_labels, new_tokens_ids = submission_helper.subtokens_to_tokens(tokenized_sentences, predictions, tokenized_sentences_ids)

submission_helper.prediction_lists_to_csv(new_tokens_ids, new_tokens, new_labels, test_csv_filepath)
