# Large Language Models
[1, Instruction fine-tuning Llama 2 with PEFT’s QLoRa method](https://github.com/nestor-sun/llm/blob/main/demo/fine-tune.py) <br/>
[2, PEFT](https://github.com/nestor-sun/llm/blob/main/demo/peft-fine-tune.py)
Parameter-efficient fine-tuning reduces the number of trainable parameters. Parameter-efficient fine-tuning (PEFT) refers to a category of techniques similar to “Finetuning I,” where only the last few layers of the network are trainable and the rest are frozen.

### Fine Tuning vs. Prompt Engineering 
1. <u>Prompt Engineering </u>: Prompt engineering is about getting the model to do what you want at inference time by providing enough context, instruction and examples **without** changing the underlying weights.
2.  <u>Fine-tuning </u>: Fine-tuning is about doing the same thing, but by directly updating the model parameters using a dataset that captures the distribution of tasks you want it to accomplish.

![ezgif-4-5843089b9b](https://github.com/nestor-sun/llm/assets/26111084/465d16cb-e037-466b-a4ca-19c8a87361f6)
credit from https://magazine.sebastianraschka.com/p/finetuning-large-language-models

3.  There are three conventional approaches outlined in the figure above.
(1) Feature-Based Approach: In the feature-based approach, we load a pretrained LLM and apply it to our target dataset. Here, we are particularly interested in generating the output embeddings for the training set, which we can use as input features to train a classification model. While this approach is particularly common for embedding-focused like BERT, we can also extract embeddings from generative GPT-style model. <br/>
(2) Finetuning I – Updating The Output Layers: Similar to the feature-based approach, we keep the parameters of the pretrained LLM frozen. We only train the newly added output layers, analogous to training a logistic regression classifier or small multilayer perceptron on the embedded features.
```
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
     num_labels=2
) 

# freeze all layers
for param in model.parameters():
    param.requires_grad = False
    
# then unfreeze the two last layers (output layers)
for param in model.pre_classifier.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True
    
# finetune model
lightning_model = CustomLightningModule(model)

trainer = L.Trainer(
    max_epochs=3,
    ...
)

trainer.fit(
  model=lightning_model,
  train_dataloaders=train_loader,
  val_dataloaders=val_loader)

# evaluate model
trainer.test(lightning_model, dataloaders=test_loader)
```
(3) Finetuning II – Updating All Layers: In practice, finetuning all layers almost always results in superior modeling performance. So, when optimizing the modeling performance, the gold standard for using pretrained LLMs is to update all layers (here referred to as finetuning II). Conceptually finetuning II is very similar to finetuning I. The only difference is that we do not freeze the parameters of the pretrained LLM but finetune them as well.

```
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
     num_labels=2
) 

# freeze layers (which we don't do here)
# for param in model.parameters():
#    param.requires_grad = False
    

# finetune model
lightning_model = LightningModel(model)

trainer = L.Trainer(
    max_epochs=3,
    ...
)

trainer.fit(
  model=lightning_model,
  train_dataloaders=train_loader,
  val_dataloaders=val_loader)

# evaluate model
trainer.test(lightning_model, dataloaders=test_loader)
```
[Interested readers can find the complete code example here (https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review).]



