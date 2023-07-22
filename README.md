# Large Language Models

## Table of Content

## Milestone Papers (this table is extracted from [Github](https://github.com/Hannibal046/Awesome-LLM))
|  Date  |       keywords       |    Institute    | Paper | Publication |    
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     Transformers     |      Google      | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)|      

Another way to categorize the papers
There are three important steps for a ChatGPT-like LLM: 
1. **Pre-training** 
2. **Instruction Tuning**
3. **Alignment**
4. 
### Base LLM

|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                                       | Model License[^1] | Data License | Data Corpus
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | -------------------------------------------------------------------------------------------------------------------------------------- | ------------- |------------- |------------------------------------------------------------ |
|      GPT 3.0      | 175B |     Decoder     |                                                                                      [api](https://openai.com/api/)                                                                                      | 2020-05 | [Paper](https://arxiv.org/pdf/2005.14165.pdf)                                                                                        | - |
|       LaMDA       | 137B |     Decoder     |                                                                                                   -                                                                                                   | 2022-01 | [Paper](https://arxiv.org/pdf/2201.08239.pdf)                                                                                           | - |
|        GLM        | 130B |     Decoder     |                                                                                [ckpt](https://github.com/THUDM/GLM-130B)                                                                                | 2022-10 | [Paper](https://arxiv.org/pdf/2210.02414.pdf)                                                                                         | [The GLM-130B License](https://github.com/THUDM/GLM-130B/blob/799837802264eb9577eb9ae12cd4bad0f355d7d6/MODEL_LICENSE) |
|        YaLM        | 100B |     Decoder     |                                                                               [ckpt](https://github.com/yandex/YaLM-100B)                                                                               | 2022-06 | [Blog](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6)     | [Apache 2.0](https://github.com/yandex/YaLM-100B/blob/14fa94df2ebbbd1864b81f13978f2bf4af270fcb/LICENSE) |
|       LLaMA       |  65B  |      Decoder      |                                                                          [ckpt](https://github.com/facebookresearch/llama) | -| -|-| Public | CommonCrawl, C4, Github, Wikipedia, etc.

### Instruction finetuned LLM
|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                              | Model License[^1] |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | ----------------------------------------------------------------------------------------------------------------------------- | ------------- |
|Flan-PaLM| 540B | Decoder |-|2022-10|[Paper](https://arxiv.org/pdf/2210.11416.pdf)| - |
|BLOOMZ| 176B | Decoder | [ckpt](https://huggingface.co/bigscience/bloomz) |2022-11|[Paper](https://arxiv.org/pdf/2211.01786.pdf)| [BigScience RAIL License v1.0](https://huggingface.co/spaces/bigscience/license) |
| InstructGPT |175B| Decoder | [api](https://platform.openai.com/overview) | 2022-03 | [Paper](https://arxiv.org/pdf/2203.02155.pdf) | - |

### RLHF LLM
|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                        |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | ----------------------------------------------------------------------------------------------------------------------------- |
| GPT 4  | - | - | - | 2023-03 | [Blog](https://openai.com/research/gpt-4)|
|      ChatGPT      |  -  |     Decoder     |                                                                                 [demo](https://openai.com/blog/chatgpt/)\|[api](https://share.hsforms.com/1u4goaXwDRKC9-x9IvKno0A4sk30)   | 2022-11 | [Blog](https://openai.com/blog/chatgpt/)      |

The above tables coule be better summarized by this wonderful visualization from this [survey paper](https://arxiv.org/abs/2304.13712):

<p align="center">
<img width="600" src="https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/survey-gif-test.gif"/>
</p>

## Other Awesome Lists
- [Pre-training]
- Pretraining data
  > **RedPajama**, 2023. [Repo](https://github.com/togethercomputer/RedPajama-Data)
  > **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**, Arxiv 2020. [Paper](https://arxiv.org/abs/2101.00027)
  > **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/)
  > **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361)
  > **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158)
  > **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, 2022. [Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) - A collection of prompt examples to be used with the ChatGPT model.
- [Prompt-Learning]
  > (2020-12) **Making Pre-trained Language Models Better Few-shot Learners** [paper](https://arxiv.org/pdf/2012.15723.pdf) <br>
  > (2021-07) **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing** [paper](https://arxiv.org/abs/2107.13586)
- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers) - A trend starts from `Natrural-Instruction` (ACL 2022), `FLAN` (ICLR 2022) and `T0` (ICLR 2022).
- [Chain-of-Thought] —a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.
  > (2021-01) **Chain of Thought Prompting Elicits Reasoning in Large Language Models.**  [paper](https://arxiv.org/abs/2201.11903)
