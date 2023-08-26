# Fine Tuning vs. Prompt Engineering 
### 1. Prompt Engineering
Prompt engineering is about getting the model to do what you want at inference time by providing enough context, instruction and examples **without** changing the underlying weights.
### 2.  Fine-tuning
Fine-tuning is about doing the same thing, but by directly updating the model parameters using a dataset that captures the distribution of tasks you want it to accomplish.

![ezgif-4-5843089b9b](https://github.com/nestor-sun/llm/assets/26111084/465d16cb-e037-466b-a4ca-19c8a87361f6)

### 3. Implementation
There are three conventional approaches outlined in the figure (credit from [Sebastian](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)) above.
#### (1) Feature-Based Approach
In the feature-based approach, we load a pretrained LLM and apply it to our target dataset. Here, we are particularly interested in generating the output embeddings for the training set, which we can use as input features to train a classification model. While this approach is particularly common for embedding-focused like BERT, we can also extract embeddings from generative GPT-style model. 
#### (2) Finetuning I 
Updating The Output Layers: Similar to the feature-based approach, we keep the parameters of the pretrained LLM frozen. We only train the newly added output layers, analogous to training a logistic regression classifier or small multilayer perceptron on the embedded features.
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
#### (3) Finetuning II 
Updating All Layers: In practice, finetuning all layers almost always results in superior modeling performance. So, when optimizing the modeling performance, the gold standard for using pretrained LLMs is to update all layers (here referred to as finetuning II). Conceptually finetuning II is very similar to finetuning I. The only difference is that we do not freeze the parameters of the pretrained LLM but finetune them as well.

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
[Interested readers can find the complete code example [here](https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review).]

## Large Language Models
### 1, Instruction fine-tuning Llama 2 with PEFT’s QLoRa [method](https://github.com/nestor-sun/llm/blob/main/demo/fine-tune.py) 
#### Prepare Your Dataset
Instruction fine-tuning is a common technique used to fine-tune a base LLM for a specific downstream use-case. The training examples look like this:
```
Below is an instruction that describes a sentiment analysis task...

### Instruction:
Analyze the following comment and classify the tone as...

### Input:
I love reading your articles...

### Response:
friendly & constructive
```
#### How to create an instruction dataset?
(1) Using an existing dataset and converting it into an instruction dataset, e.g., [FLAN](https://huggingface.co/datasets/SirNeural/flan_v2)<br/>
(2) Use existing LLMs to create synthetically instruction datasets, e.g., [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) <br/>
(3) Use Humans to create instructions datasets, e.g., [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

### 2, [PEFT](https://github.com/nestor-sun/llm/blob/main/demo/peft-fine-tune.py)
Parameter-efficient finetuning allows us to reuse pretrained models while minimizing the computational and resource footprints. In sum, parameter-efficient finetuning is useful for at least 5 reasons:

Reduced computational costs (requires fewer GPUs and GPU time);

Faster training times (finishes training faster);

Lower hardware requirements (works with smaller GPUs & less smemory);

Better modeling performance (reduces overfitting);

Less storage (majority of weights can be shared across different tasks).

3, Reinforcement Learning with Human Feedback: In Reinforcement Learning with Human Feedback (RLHF), a pretrained model is finetuned using a combination of supervised learning and reinforcement learning -- the approach was popularized by the original ChatGPT model, which was in turn based on InstructGPT ([Ouyang et al.](https://arxiv.org/abs/2203.02155)). 
In RLHF, human feedback is collected by having humans rank or rate different model outputs, providing a reward signal. The collected reward labels can then be used to train a reward model that is then in turn used to guide the LLMs adaptation to human preferences. The reward model itself is learned via supervised learning (typically using a pretrained LLM as base model). Next, the reward model is used to update the pretrained LLM that is to be adapted to human preferences -- the training uses a flavor of reinforcement learning called proximal policy optimization ([Schulman et al.](https://arxiv.org/abs/1707.06347)).
![7dfa415c-da9c-4d6f-8de8-ffc9f92272db_1602x952](https://github.com/nestor-sun/llm/assets/26111084/f2081679-dadc-4811-8b90-9d01f5a02c18)
Screenshot from the InstructGPT [paper](https://arxiv.org/pdf/2203.02155.pdf) outlining the RLHF process.

