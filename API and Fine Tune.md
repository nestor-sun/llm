# Large Language Models
[1, Instruction fine-tuning Llama 2 with PEFT’s QLoRa method](https://github.com/nestor-sun/llm/blob/main/demo/fine-tune.py) <br/>
[2, PEFT](https://github.com/nestor-sun/llm/blob/main/demo/peft-fine-tune.py)
Parameter-efficient fine-tuning reduces the number of trainable parameters. Parameter-efficient fine-tuning (PEFT) refers to a category of techniques similar to “Finetuning I,” where only the last few layers of the network are trainable and the rest are frozen.

### Fine Tuning vs. Prompt Engineering 
1. <u>Prompt Engineering </u>: Prompt engineering is about getting the model to do what you want at inference time by providing enough context, instruction and examples **without** changing the underlying weights.
