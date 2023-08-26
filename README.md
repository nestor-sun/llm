# Large Language Models

## Table of Content

## Milestone Papers (credit from [Github](https://github.com/Hannibal046/Awesome-LLM))
|  Date  |       keywords       |    Institute    | Paper | Publication |    
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     Transformers     |      Google      | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)| NeurIPS 2017|
| 2018-06 |       GPT 1.0       |      OpenAI      | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | OpenAI|
| 2018-10 |         BERT         |      Google      | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)| NAACL 2019|
| 2019-02 |       GPT 2.0       |      OpenAI      | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)| - |
| 2019-10 |          T5          |      Google      | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)| JMLR 2020 |
| 2020-01 |     Scaling Law     |      OpenAI      | [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)|-|
| 2020-05 |       GPT 3.0       |      OpenAI      | [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) | NeurIPS 2020| 
| 2021-01 | Switch Transformers |      Google      | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)| JMLR 2022|
| 2021-09 |         FLAN         |      Google      | [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR) | ICLR 2022|
| 2021-12 |        Retro        |     DeepMind     | [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)|ICML 2022|
| 2022-01 |        LaMDA        |      Google      | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)|-|
| 2022-01 | Megatron-Turing NLG | Microsoft&NVIDIA | [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf)|-|
| 2022-03 |     InstructGPT     |      OpenAI      | [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)|-|
| 2022-05 |         OPT         |       Meta       | [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)|-|
| 2022-06 |  Emergent Abilities  |      Google      | [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)|TMLR 2022|
| 2022-06 |        METALM        |    Microsoft    | [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)|-|
| 2022-10 |       GLM-130B       |     Tsinghua     | [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/pdf/2210.02414.pdf)|-|
| 2022-11 |         HELM         |     Stanford     | [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110.pdf) |-|
| 2023-02 | LLaMA|Meta|[LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)|-|
| 2023-03 | GPT 4 | OpenAI | [GPT-4 Technical Report](https://openai.com/research/gpt-4)|-|

## Timeline of LLMs
![LLMs_timeline](https://github.com/RUCAIBox/LLMSurvey/blob/main/assets/LLMs-0623-final.png)

<br/>
The wonderful visualization below (from this that summarizes the evolutionary tree of modern LLMs) traces the development of language models in recent years and highlights some of the most well-known models: [survey paper](https://arxiv.org/pdf/2304.13712.pdf)
<p align="center">
<img width="600" src="https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/survey-gif-test.gif"/>
</p>




## Other Awesome Lists
- Pretraining data
  > **RedPajama**, 2023. [Repo](https://github.com/togethercomputer/RedPajama-Data) <br>
  > **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**, Arxiv 2020. [Paper](https://arxiv.org/abs/2101.00027) <br>
  > **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/) <br>
  > **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361) <br>
  > **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158) <br>
  > **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, 2022. [Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts): a collection of prompt examples to be used with the ChatGPT model.
- Prompt-Learning
  > (2020-12) **Making Pre-trained Language Models Better Few-shot Learners** [paper](https://arxiv.org/pdf/2012.15723.pdf) <br>
  > (2021-07) **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing** [paper](https://arxiv.org/abs/2107.13586)

- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers): a trend starts from `Natrural-Instruction` (ACL 2022), `FLAN` (ICLR 2022) and `T0` (ICLR 2022).
- Chain-of-Thought: a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.
  > (2021-01) **Chain of Thought Prompting Elicits Reasoning in Large Language Models.**  [paper](https://arxiv.org/abs/2201.11903)


### List of LLMs

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" align="center" rowspan="2">Category</th>
    <th class="tg-baqh" align="center" rowspan="2">model</th>
    <th class="tg-0lax" align="center" rowspan="2">Release Time</th>
    <th class="tg-baqh" align="center" rowspan="2">Size(B)</th>
    <th class="tg-0lax" align="center" rowspan="2">Link</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" align="center" rowspan="27">Publicly <br>Accessbile</td>
    <td class="tg-baqh" align="center">T5</td>
    <td class="tg-0lax" align="center">2019/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/1910.10683">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">mT5</td>
    <td class="tg-0lax" align="center">2021/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2010.11934">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PanGu-α</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2104.12369">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CPM-2</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">198</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2106.10715">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">T0</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2110.08207">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-NeoX-20B</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.06745">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CodeGen</td>
    <td class="tg-0lax" align="center">2022/03</td>
    <td class="tg-baqh" align="center">16</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2203.13474">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Tk-Instruct</td>
    <td class="tg-0lax" align="center">2022/04</td>
    <td class="tg-baqh" align="center" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.07705">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">UL2</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2205.05131">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">OPT</td>
    <td class="tg-0lax" align="center">2022/05</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2205.01068">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">YaLM</td>
    <td class="tg-0lax" align="center">2022/06</td>
    <td class="tg-baqh" align="center">100</td>
    <td class="tg-0lax" align="center"><a href="https://github.com/yandex/YaLM-100B">GitHub</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">NLLB</td>
    <td class="tg-0lax" align="center">2022/07</td>
    <td class="tg-baqh" align="center">55</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2207.04672">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">BLOOM</td>
    <td class="tg-0lax" align="center">2022/07</td>
    <td class="tg-baqh" align="center">176</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.05100">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GLM</td>
    <td class="tg-0lax" align="center">2022/08</td>
    <td class="tg-baqh" align="center">130</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.02414">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-T5</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">mT0</td>
    <td class="tg-0lax" align="center">2022/11</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.01786">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Galatica</td>
    <td class="tg-0lax" align="center" align="center" align="center">2022/11</td>
    <td class="tg-baqh" align="center" align="center">120</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.09085">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">BLOOMZ</td>
    <td class="tg-0lax" align="center">2022/11</td>
    <td class="tg-baqh" align="center">176</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.01786">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">OPT-IML</td>
    <td class="tg-0lax" align="center">2022/12</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2212.12017">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Pythia</td>
    <td class="tg-0lax" align="center">2023/01</td>
    <td class="tg-baqh" align="center">12</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2304.01373">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">LLaMA</td>
    <td class="tg-0lax" align="center">2023/02</td>
    <td class="tg-baqh" align="center">7/13/65</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2302.13971v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Vicuna</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://lmsys.org/blog/2023-03-30-vicuna/">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ChatGLM</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">6</td>
    <td class="tg-0lax" align="center"><a href="https://github.com/THUDM/ChatGLM-6B">GitHub</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CodeGeeX</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2303.17568">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Koala</td>
    <td class="tg-0lax" align="center">2023/04</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://bair.berkeley.edu/blog/2023/04/03/koala/">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Falcon</td>
    <td class="tg-0lax" align="center">2023/06</td>
    <td class="tg-baqh" align="center">7/40</td>
    <td class="tg-0lax" align="center"><a href="https://huggingface.co/docs/transformers/main/model_doc/falcon">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Llama-2</td>
    <td class="tg-0lax" align="center">2023/07</td>
    <td class="tg-baqh" align="center">7/13/70</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/pdf/2307.09288.pdf">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-nrix" align="center" rowspan="31">Closed<br>Source</td>
    <td class="tg-baqh" align="center">GShard</td>
    <td class="tg-0lax" align="center">2020/01</td>
    <td class="tg-baqh" align="center" align="center">600</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2006.16668v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-3</td>
    <td class="tg-0lax" align="center">2020/05</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2005.14165">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">LaMDA</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">137</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2201.08239">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">HyperCLOVA</td>
    <td class="tg-0lax" align="center">2021/06</td>
    <td class="tg-baqh" align="center">82</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2109.04650">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Codex</td>
    <td class="tg-0lax" align="center">2021/07</td>
    <td class="tg-baqh" align="center">12</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2107.03374">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ERNIE 3.0</td>
    <td class="tg-0lax" align="center" align="center">2021/07</td>
    <td class="tg-baqh" align="center">10</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2107.02137">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Jurassic-1</td>
    <td class="tg-0lax" align="center">2021/08</td>
    <td class="tg-baqh" align="center">178</td>
    <td class="tg-0lax" align="center"><a href="https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center" align="center">FLAN</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">137</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2109.01652">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">MT-NLG</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">530</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2201.11990">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Yuan 1.0</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">245</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2110.04725">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Anthropic</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">52</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.00861">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">WebGPT</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.09332">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Gopher</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">280</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2112.11446v2">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ERNIE 3.0 Titan</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">260</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.12731">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GLaM</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">1200</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.06905">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">InstructGPT</td>
    <td class="tg-0lax" align="center">2022/01</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2203.02155v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">AlphaCode</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">41</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2203.07814v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Chinchilla</td>
    <td class="tg-0lax" align="center">2022/03</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2203.15556">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PaLM</td>
    <td class="tg-0lax" align="center">2022/04</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.02311">Paper</a></td>
    <tr>
    <td class="tg-baqh" align="center">Cohere</td>
    <td class="tg-0lax" align="center">2022/06</td>
    <td class="tg-baqh" align="center">54</td>
    <td class="tg-0lax" align="center"><a href="https://cohere.ai/">Homepage</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">AlexaTM</td>
    <td class="tg-0lax" align="center">2022/08</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2208.01448">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Luminous</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="https://docs.aleph-alpha.com/docs/introduction/luminous/">Docs</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Sparrow</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2209.14375v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">WeLM</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">10</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2209.10372">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">U-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11399">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-U-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Alpaca</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">7</td>
    <td class="tg-0lax" align="center"><a href="https://crfm.stanford.edu/2023/03/13/alpaca.html">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-4</td>
    <td class="tg-0lax" align="center">2023/3</td>
    <td class="tg-baqh" align="center">-</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2303.08774v2">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PanGU-Σ</td>
    <td class="tg-0lax" align="center">2023/3</td>
    <td class="tg-baqh" align="center">1085</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2303.10845">Paper</a></td>
  </tr>
</tbody>
</table>

### Commonly Used Corpora

1. <u>BookCorpus</u>: **"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books"**. *Yukun Zhu et al.*  ICCV 2015. [[Paper](http://arxiv.org/abs/1506.06724v1)] [[Source](https://huggingface.co/datasets/bookcorpus)]
2. <u>Guntenburg</u>: [[Source](https://www.gutenberg.org/)]
3. <u>CommonCrawl</u>: [[Source](https://commoncrawl.org/)]
4. <u>C4</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](http://arxiv.org/abs/1910.10683v3)] [[Source](https://www.tensorflow.org/datasets/catalog/c4)]
5. <u>CC-stories-R</u>: **"A Simple Method for Commonsense Reasoning"**. *Trieu H. Trinh el al.* arXiv 2018. [[Paper](http://arxiv.org/abs/1806.02847v2)] [[Source](https://huggingface.co/datasets/spacemanidol/cc-stories)]
6. <u>CC-NEWS</u>: **"RoBERTa: A Robustly Optimized BERT Pretraining Approach"**. *Yinhan Liu et al.* arXiv 2019. [[Paper](http://arxiv.org/abs/1907.11692v1)] [[Source](https://huggingface.co/datasets/cc_news)]
7. <u>REALNEWs</u>: **"Defending Against Neural Fake News"**. *Rowan Zellers et al.* NeurIPS 2019. [[Paper](http://arxiv.org/abs/1905.12616v3)] [[Source](https://github.com/rowanz/grover/tree/master/realnews)]
8. <u>OpenWebText</u>: [[Source](https://skylion007.github.io/OpenWebTextCorpus/)]
9. <u>Pushshift.io</u>: **"The Pushshift Reddit Dataset"**. *Jason Baumgartner et al*. AAAI 2020. [[Paper](http://arxiv.org/abs/2001.08435v1)] [[Source](https://files.pushshift.io/reddit/)]
10. <u>Wikipedia</u>: [[Source](https://dumps.wikimedia.org/)]
11. <u>BigQuery</u>:  [[Source](https://cloud.google.com/bigquery/public-data?hl=zh-cn)]
12. <u>The Pile</u>: **"The Pile: An 800GB Dataset of Diverse Text for Language Modeling"**. *Leo Gao et al*. arxiv 2021. [[Paper](http://arxiv.org/abs/2101.00027v1)] [[Source](https://pile.eleuther.ai/)]
13. <u>ROOTS</u>: **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]


