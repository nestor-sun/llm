# Large Language Models
[1, Instruction fine-tuning Llama 2 with PEFT’s QLoRa method](https://github.com/nestor-sun/llm/blob/main/demo/fine-tune.py) <br/>
[2, PEFT](https://github.com/nestor-sun/llm/blob/main/demo/peft-fine-tune.py)
Parameter-efficient fine-tuning reduces the number of trainable parameters. Parameter-efficient fine-tuning (PEFT) refers to a category of techniques similar to “Finetuning I,” where only the last few layers of the network are trainable and the rest are frozen.

### Fine Tuning vs. Prompt Engineering 
1. <u>Prompt Engineering </u>: Prompt engineering is about getting the model to do what you want at inference time by providing enough context, instruction and examples **without** changing the underlying weights.
2.  <u>Fine-tuning </u>: Fine-tuning is about doing the same thing, but by directly updating the model parameters using a dataset that captures the distribution of tasks you want it to accomplish.

![ezgif-4-5843089b9b](https://github.com/nestor-sun/llm/assets/26111084/465d16cb-e037-466b-a4ca-19c8a87361f6)
3.  There are three conventional approaches outlined in the figure above.
(1) Feature-Based Approach: In the feature-based approach, we load a pretrained LLM and apply it to our target dataset. Here, we are particularly interested in generating the output embeddings for the training set, which we can use as input features to train a classification model. While this approach is particularly common for embedding-focused like BERT, we can also extract embeddings from generative GPT-style model. <br/>
(2) Finetuning I – Updating The Output Layers: 





