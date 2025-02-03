# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kC4lXumwec0dw_VEHBEDgxPIkaxiuI69
"""

!pip install transformers
import torch
#from longformer.longformer import Longformer, LongformerConfig # This is deprecated
#from longformer.sliding_chunks import pad_to_window_size # This is deprecated
!pip install datasets transformers
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
!pip install scikit-learn
from transformers import EarlyStoppingCallback
import matplotlib.pyplot as plt
import torch
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
from transformers import LongformerModel, LongformerConfig
from transformers import LongformerTokenizer, LongformerForSequenceClassification
# model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# The above line downloads the model using 'transformers' package, so there's no need for the separate 'longformer' package.

"""tokenization"""

# Load the CSV files
train_df = pd.read_csv('/content/dataset_train_updated.csv')
test_df = pd.read_csv('/content/dataset_test_updated.csv')

# Initialize the tokenizer and model for Longformer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=len(train_df['class'].unique()))

# Tokenization function for Longformer
def tokenize_function(examples):
    return tokenizer(
        examples['text'],  # Replace 'text' with the appropriate column name in your dataframe if different
        padding="max_length",
        truncation=True,
        max_length=4096,  # Longformer's max token limit
        return_tensors="pt"  # Return PyTorch tensors
    )

"""training"""

# Split train_df into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['class'], test_size=0.1, random_state=42
)
# Create DataFrame for validation set
val_df = pd.DataFrame({'text': val_texts, 'class': val_labels})
# Convert to Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)
# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename 'class' to 'labels' explicitly
if 'class' in train_dataset.column_names:
    train_dataset = train_dataset.rename_column('class', 'labels')
if 'class' in val_dataset.column_names:
    val_dataset = val_dataset.rename_column('class', 'labels')
if 'class' in test_dataset.column_names:
    test_dataset = test_dataset.rename_column('class', 'labels')


# Convert to PyTorch format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments with mini-batching (gradient accumulation)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    remove_unused_columns=False
)
# Initialize the EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3  # Stop training after 3 evaluation steps with no improvement
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Validation dataset
    callbacks=[early_stopping_callback]
)

# Train the model
torch.cuda.empty_cache()
trainer.train()

# Save the trained model
model.save_pretrained("longformer_finetuned")

# Save the tokenizer as well
tokenizer.save_pretrained("longformer_finetuned")

"""test"""

# Make predictions on the test dataset
test_predictions = trainer.predict(test_dataset)

# Get the predicted labels (class) from the logits
predicted_labels = torch.argmax(torch.tensor(test_predictions.predictions), axis=-1)

# Add the predictions to the test dataframe
test_df['predicted_class'] = predicted_labels.numpy()

# Calculate accuracy
true_labels = test_df['class'].values
accuracy = accuracy_score(true_labels, predicted_labels.numpy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels.numpy())

# Display the confusion matrix using a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Dementia", "Control"], yticklabels=["Dementia", "Control"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the predictions with accuracy
test_df.to_csv('dataset_test_predictions.csv', index=False)

print("Model training complete, and predictions saved to dataset_test_predictions.csv")

