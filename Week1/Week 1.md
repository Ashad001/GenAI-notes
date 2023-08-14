# ***Transformers, Project Life cycle and Hyper

parameters***

- [\*\*\*Transformers, Project Life cycle and Hyper](#transformers-project-life-cycle-and-hyper)
- [**Text Generation before Transformers**](#text-generation-before-transformers)
  - [**RNNs**](#rnns)
- [**Transformers**](#transformers)
  - [Transformers Architecture](#transformers-architecture)
  - [Generating Text using Transformers](#generating-text-using-transformers)
  - [Summary](#summary)
- [Prompts and Prompt Engineering](#prompts-and-prompt-engineering)
  - [In-context Learning](#in-context-learning)
    - [Zero hot Inference](#zero-hot-inference)
    - [One Hot Inference](#one-hot-inference)
    - [Few Shot Inference](#few-shot-inference)
- [Generative Configurations](#generative-configurations)
  - [Max New Tokens](#max-new-tokens)
  - [Greedy vs Random Sampling](#greedy-vs-random-sampling)
    - [Top `p` and Top `k`](#top-p-and-top-k)
  - [Temperature](#temperature)
- [Generative AI project Lifecycle](#generative-ai-project-lifecycle)
  - [Scoping](#scoping)
  - [Select](#select)
  - [Assessing Model Performance and Refinement](#assessing-model-performance-and-refinement)
  - [Model Deployment and Infrastructure Integration](#model-deployment-and-infrastructure-integration)
- [The Code Till Now](#the-code-till-now)
  - [**Generative AI Use Case: Summarize Dialogue**](#generative-ai-use-case-summarize-dialogue)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Summarize Dialogue without Prompt Engineering](#summarize-dialogue-without-prompt-engineering)
    - [Zero Shot Inference with an Instruction Prompt](#zero-shot-inference-with-an-instruction-prompt)
    - [Zero Shot Inference with the Prompt Template from FLAN-T5](#zero-shot-inference-with-the-prompt-template-from-flan-t5)
  - [Summarize Dialogue with One Shot and Few Shot Inference](#summarize-dialogue-with-one-shot-and-few-shot-inference)
    - [One Shot Inference](#one-shot-inference)
    - [Few Shot Inference](#few-shot-inference-1)
  - [Generative Configuration Parameters for Inference](#generative-configuration-parameters-for-inference)
- [Pre-training Large Language Models](#pre-training-large-language-models)
  - [Auto-encoding Models (Encoder only)](#auto-encoding-models-encoder-only)
  - [Autoregressive Model (Decoder Only)](#autoregressive-model-decoder-only)
    - [Sequence to Sequence Modeling](#sequence-to-sequence-modeling)
- [Computational Challengers of training LLMs](#computational-challengers-of-training-llms)
  - [Quantization](#quantization)
    - [FP16](#fp16)
    - [BFLOAT16 | BF16](#bfloat16--bf16)
    - [INT8](#int8)
    - [Summary](#summary-1)
- [BloombergGPT](#bloomberggpt)
- [Quiz](#quiz)
  - [Reading Resources](#reading-resources)
    - [**Transformer Architecture**](#transformer-architecture)
    - [**Pre-training and scaling laws**](#pre-training-and-scaling-laws)
    - [**Model architectures and pre-training objectives**](#model-architectures-and-pre-training-objectives)
    - [**Scaling laws and compute-optimal models**](#scaling-laws-and-compute-optimal-models)
- [**Acknowledgements**](#acknowledgements)

# **Text Generation before Transformers**

It's important to note that generative algorithms are not new. Previous generations of language models made use of an architecture called recurrent neural networks or RNNs.

## **RNNs**

RNNs while powerful for their time, were limited by the amount of compute and memory needed to perform well at generative tasks.

The prediction's quality is limited with just a single previous word visible to the model. Increasing the RNN's capability to consider more preceding words requires substantial resource expansion. Despite scaling up the model, accurate prediction remains elusive due to inadequate input exposure. Effective next-word prediction demands a broader context beyond a handful of prior words—ideally encompassing the entire sentence or document.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%201.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%202.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%203.png)

<aside>
💡 The challenge lies in the intricacies of language. Numerous languages feature words with multiple interpretations, known as homonyms. Context within a sentence clarifies the intended meaning

</aside>

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%204.png)

such as, distinguishing between types of "bank." Words positioned in sentence structures can introduce ambiguity or what we term as syntactic ambiguity.

# **Transformers**

<aside>
💡 In 2017, the "Attention is All You Need" paper by Google and the University of Toronto introduced the transformer architecture. This approach revolutionized generative AI, enabling efficient scaling with multi-core GPUs, parallel processing of larger datasets, and a focus on understanding word meaning through attention. The title encapsulates its essence: "attention is all you need.”

</aside>

<aside>
💡 This article is considered to be the start of generative AI:

</aside>

[Attention is All You Need – Google Research](https://research.google/pubs/pub46201/)

## Transformers Architecture

The transformer architecture excels at understanding the meaning and context of all words in a sentence. It doesn't just consider nearby words but includes every word. Through attention weights, the model captures how each word relates to others, regardless of their position. This enables the algorithm to identify the book's owner, possible possessors, and its relevance within the overall document context.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%205.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%206.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%207.png)

Attention weights are acquired during Language Model (LLM) training. The visual representation, known as an attention map, depicts the weights between each word and all others. In this simplified instance, "book" prominently relates to "teacher" and "student." This self-attention mechanism, fostering comprehension across the entire input, greatly enhances the model's language encoding proficiency.

The transformer architecture comprises two distinct segments: the encoder and the decoder. These elements collaborate and exhibit several commonalities. Observe the model's input located at the bottom, while outputs emerge at the top.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%208.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%209.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2010.png)

Machine-learning models are essentially advanced statistical calculators that work with numbers, not words. Texts must be tokenized before input, converting words into numbers based on a predefined dictionary. Various tokenization methods exist, like full-word token IDs or partial word representations. Consistency in tokenizer use is crucial between training and text generation.

Words become numbers using tokenization, where each number corresponds to a word's position in a model-friendly dictionary. Tokenization methods vary, such as matching IDs to words or parts of words. Consistency between training and text generation with the chosen tokenizer is key.

Once text turns numeric, it enters the embedding layer—a trainable vector space. Here, tokens become distinct vectors, encoding meaning and context within the input sequence.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2011.png)

![See how student and book relate/close to each other.](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2012.png)

See how student and book relate/close to each other.

Embedding vector spaces have long been used in language processing. Think of it like assigning each word a unique spot in a mathematical space. In our example, each word gets a number, and this number connects to a point in this space. **Even though the real space is larger (like 512 dimensions)**, imagine a simple three-dimensional version. Words close in this space are related, and you can measure their "distance" as an angle. This mathematical approach helps the model understand language better.

When inserting token vectors into the encoder or decoder, you also introduce positional encoding. This maintains word order information and the word's position significance within the sentence. With both input tokens and positional encodings summed, the resulting vectors move to the self-attention layer.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2013.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2014.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2015.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2016.png)

In this step, the model examines how tokens in your input sequence interact. This enables the model to focus on various segments of the input to capture word connections more effectively. The self-attention weights learned during training and stored here indicate the significance of each word in the input sequence concerning all other words.

However, this process isn't singular; the transformer architecture employs multi-headed self-attention. This signifies that multiple sets of self-attention weights or "heads" are learned concurrently, each operating independently. The quantity of attention heads in the layer can differ between models, often falling within the range of 12 to 100.

The idea is that each self-attention head grasps a unique language aspect. For instance, one head might understand relationships between people in a sentence, while another concentrates on the sentence's action. Additional heads might even consider attributes like word rhyme.

## Generating Text using Transformers

In this instance, we'll explore a translation task, a fundamental goal of the transformer architecture creators. We'll employ a transformer model to convert the French phrase `J'adore l'apprentissage automatique.`into English. Initially, we'll tokenize the input words using the same tokenizer employed for network training.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2017.png)

After tokenizing, these units join the encoder's input. They journey through the embedding and multi-headed attention layers. The results then pass through a feed-forward network and exit the encoder. This departing data holds a comprehensive understanding of the input's structure and meaning.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2018.png)

This representation is integrated into the decoder's core to impact its self-attention mechanisms. Subsequently, a "start of sequence" token is introduced to the decoder's input. This signals the decoder to forecast the subsequent token, drawing from the contextual comprehension furnished by the encoder.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2019.png)

The output of the decoder's self-attention layers gets passed through the decoder feed-forward network and through a final softmax output layer. At this point, we have our first token. You'll continue this loop, passing the output token back to the input to trigger the generation of the next token, until the model predicts an end-of-sequence token.

At this point, the final sequence of tokens can be detokenized into words, and you have your output. In this case, I love machine learning.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2020.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2021.png)

## Summary

In simple terms, the topic is about building large language models using a special architecture called the transformer. This architecture allows the model to understand the relevance and context of all the words in a sentence, not just the neighboring words. It does this by applying attention weights to the relationships between each word and every other word in the sentence. This helps the model learn the meaning and connections between words, even if they are far apart in the input.

Here are the key points in simpler terms:

- The transformer architecture improves the performance of natural language tasks.
- It uses attention weights to understand the relevance of each word to every other word in a sentence.
- This ability to learn relationships across the whole input significantly improves the model's ability to encode language.
- The transformer architecture has two parts: the encoder and the decoder.
- Before passing text into the model, the words are converted into numbers using a process called tokenization.
- The tokenized words are then passed through an embedding layer, which represents each word as a vector in a high-dimensional space.
- The model also adds positional encoding to preserve the word order and relevance in the sentence.
- The self-attention layer analyzes the relationships between the tokens in the input sequence.
- The transformer architecture has multiple sets of self-attention weights, called attention heads, which learn different aspects of language.
- The output of the model is a probability score for each word in the vocabulary, indicating the likelihood of that word being the next word in the sequence.

# Prompts and Prompt Engineering

The input text is termed the "prompt," while the process of creating text is "inference," and the resulting text is the "completion." The extent of text available for the prompt is the "context window." While the displayed example exhibits successful performance, you may encounter instances where the model doesn't initially provide the desired output. Adjusting the prompt's language or structure might require multiple attempts to achieve the desired behavior.

## In-context Learning

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2022.png)

A potent technique to enhance model outcomes is incorporating task-related examples within the prompt. This practice, known as in-context learning, involves including task-related examples or extra data within the context window. In-context learning empowers language models to better comprehend the intended task.

### Zero hot Inference

In the provided prompt, you instruct the model to assess the sentiment of a review—whether it's positive or negative. The prompt comprises the directive "Classify this review," followed by contextual content (the review text), and an instruction for generating the sentiment conclusion.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2023.png)

This approach, involving your input data within the prompt, is referred to as zero-shot inference. Remarkably, the largest language models excel at this, comprehending the task and delivering accurate responses.

### One Hot Inference

Including an example in the prompt boosts performance. The extended prompt starts with a complete task example. After instructing the model to classify a review, it shows a positive review ("I loved this movie"). Then, the actual input review is presented.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2024.png)

This richer prompt helps the smaller model better grasp the task and response format, known as one-shot inference.

### Few Shot Inference

For deeper learning, you can use multiple examples, termed few-shot inference. With a smaller model struggling in one-shot inference, you're trying few-shot with a second example—a negative review. Presenting these prompts, the model now grasps the instruction, accurately identifying the review's negative sentiment.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2025.png)

This mix of examples aids the model's understanding.

# Generative Configurations

- LLMs, like those on Hugging Face and AWS, come with controls to adjust their behavior.
- Configuration parameters influence LLM output during inference.
- These parameters are distinct from training parameters and impact aspects like token count and output creativity.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2026.png)

## Max New Tokens

- "Max new tokens" is a simple parameter to limit the generated tokens.
- It sets a cap on the selection process iterations during token generation.
- Examples: Setting max new tokens to 100, 150, or 200.
- Completion length can vary due to different stop conditions, like predicting an end of sequence token.
- Remember it's max new tokens, not a hard number of new tokens generated.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2027.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2028.png)

## Greedy vs Random Sampling

- Default operation of large language models involves "greedy decoding."
- Greedy decoding is a basic next-word prediction method.
- It always selects the word with the highest probability.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2029.png)

- Effective for short generations, but prone to repeated words or sequences.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2030.png)

- Reduces word repetition, but can lead to overly creative or nonsensical outputs.
- Possibility of wandering off-topic or generating nonsensical words based on settings.

- Random sampling introduces variability in language model outputs.
- Instead of always choosing the most probable word, random sampling selects based on probability distribution.
- Example: Word "banana" with a probability score of 0.02 leads to a 2% chance of selection.
- In some cases, you may need to explicitly disable greedy decoding and enable random sampling.
- Example: Hugging Face transformers implementation requires setting `do_sample` to "true."
- Enabling random sampling introduces variability in outputs, reducing repetition.
- This adjustment can enhance the model's creativity and output quality.

### Top `p` and Top `k`

- "Top-p" and "top-k" are sampling techniques to refine random sampling.
- They enhance the likelihood of producing sensible outputs.
- "Top-p" filters out words until a cumulative probability (p) is reached, controlling output randomness.
- "Top-k" selects from the top-k most probable words, reducing randomness and enhancing coherence.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2031.png)

## Temperature

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2032.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2033.png)

- "Temperature" is a parameter controlling model output randomness.
- It affects the probability distribution for the next token.
- Higher temperature increases randomness; lower temperature reduces randomness.
- Temperature is a scaling factor applied in the final softmax layer.
- Influences the shape of the probability distribution for the next token.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2034.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2035.png)

- Increased randomness with higher temperature leads to more variable output.
- High temperature promotes creativity in text generation.

- Probability concentration in this example, Word "cake" with high likelihood in distribution.
- Lower temperature (close to 0) leads to less random output, following learned sequences.
- Higher temperature (greater than 1) results in a flatter, broader probability distribution.

<aside>
💡 If you leave the temperature value equal to one, this will leave the SoftMax function as default

</aside>

# Generative AI project Lifecycle

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2036.png)

This topic discusses the techniques needed for creating and implementing an LLM-based application.

It presents a generative AI project life cycle as a guide from idea inception to launch.

The framework outlines the steps necessary to move your project from concept to completion.

## Scoping

- Defining scope is a crucial initial step in any project.
- LLM capabilities vary based on model size and architecture.
- Determine the LLM's function within your application.
- Consider broad tasks vs. specific functions (e.g., long-form generation vs. named entity recognition).
- Specificity in defining tasks can save time and reduce compute costs.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2037.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2038.png)

## Select

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2039.png)

- After scoping, decide between building from scratch or using an existing model.
- Starting with an existing model is common.
- Some situations might warrant training a new model.
- Weigh pros and cons based on project requirements and resources.

## Assessing Model Performance and Refinement

- After scoping, assess model performance for your application.
- In-context learning, like prompt engineering, can enhance performance.
- For cases where model performance falls short, consider fine-tuning.
- Ensuring models align with human preferences in deployment is vital.
- Evaluation is crucial; next week, you'll learn metrics and benchmarks.
- Adaptation and alignment are iterative stages in development.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2040.png)

- Steps include trying prompt engineering, fine-tuning, and reevaluation.

## Model Deployment and Infrastructure Integration

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2041.png)

- Deploy the model and integrate it with your application's infrastructure.
- Optimize the model for deployment to maximize resource utilization.
- Prioritize a seamless user experience for application users.
- Account for additional infrastructure needs to ensure optimal functionality.

# The Code Till Now

Welcome to the hands-on portion. In this portion, you will see in the dialogue summarization task using generative AI.  Objective is to examine how the input text influences the model's output and practice prompt engineering to guide it effectively for your desired task. Through comparisons of zero shot, one shot, and few shot inferences.

<aside>
💡 You can run all this in google colab for FREE!

</aside>

## **Generative AI Use Case: Summarize Dialogue**

## Installation

```python
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 \
    datasets==2.11.0  --quiet
```

## Dataset

Load the datasets, Large Language Model (LLM), tokenizer, and configurator.

```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
```

## Summarize Dialogue without Prompt Engineering

In this use case, you will be generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face. The list of available models in the Hugging Face `transformers` package can be found [here](https://huggingface.co/docs/transformers/index).

Let's upload some simple dialogues from the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) Hugging Face dataset. This dataset contains 10,000+ dialogues with the corresponding manually labeled summaries and topics.

```python
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2042.png)

print couple of examples

```python

example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2043.png)

Load the [FLAN-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5), creating an instance of the `AutoModelForSeq2SeqLM` class with the `.from_pretrained()` method.

```python
model_name='google/flan-t5-base'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

To perform encoding and decoding, you need to work with text in a tokenized form. **Tokenization** is the process of splitting texts into smaller units that can be processed by the LLM models.

Download the tokenizer for the FLAN-T5 model using `AutoTokenizer.from_pretrained()` method. Parameter `use_fast` switches on fast tokenizer. At this stage, there is no need to go into the details of that, but you can find the tokenizer parameters in the [documentation](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/auto#transformers.AutoTokenizer).

```python
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

Test the tokenizer encoding and decoding a simple sentence:

```python
sentence = "What time is it, Tom?"

sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0], 
        skip_special_tokens=True
    )

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2044.png)

Now it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering. **Prompt engineering** is an act of a human changing the **prompt** (input) to improve the response for a given task.

```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')
```

You can see that the guesses of the model make some sense, but it doesn't seem to be sure what task it is supposed to accomplish. Seems it just makes up the next sentence in the dialogue.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2045.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2046.png)

Prompt engineering is an important concept in using foundation models for text generation. You can check out [this blog](https://www.amazon.science/blog/emnlp-prompt-engineering-is-the-new-feature-engineering) from Amazon Science for a quick introduction to prompt engineering.

### Zero Shot Inference with an Instruction Prompt

In order to instruct the model to perform a task - summarize a dialogue - you can take the dialogue and convert it into an instruction prompt. This is often called **zero shot inference**. You can check out [this blog from AWS](https://aws.amazon.com/blogs/machine-learning/zero-shot-prompting-for-the-flan-t5-foundation-model-in-amazon-sagemaker-jumpstart/) for a quick description of what zero shot learning is and why it is an important concept to the LLM model.

Wrap the dialogue in a descriptive instruction and see how the generated text will change:

```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
    """

    # Input constructed prompt instead of the dialogue.
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)    
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2047.png)

### Zero Shot Inference with the Prompt Template from FLAN-T5

Let's use a slightly different prompt. FLAN-T5 has many prompt templates that are published for certain tasks [here](https://github.com/google-research/FLAN/tree/main/flan/v2). In the following code, you will use one of the [pre-built FLAN-T5 prompts](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py):

```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
        
    prompt = f"""
Dialogue:

{dialogue}

What was going on?
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2048.png)

## Summarize Dialogue with One Shot and Few Shot Inference

**One shot and few shot inference** are the practices of providing an LLM with either one or more full examples of prompt-response pairs that match your task - before your actual prompt that you want completed. This is called "in-context learning" and puts your model into a state that understands your specific task. You can read more about it in [this blog from HuggingFace](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api).

### One Shot Inference

Let's build a function that takes a list of `example_indices_full`, generates a prompt with full examples, then at the end appends the prompt which you want the model to complete (`example_index_to_summarize`). You will use the same FLAN-T5 prompt template from section [3.2](https://d-ayeptnmatpko.studio.us-east-1.sagemaker.aws/jupyter/default/lab/tree/Lab_1_summarize_dialogue.ipynb#3.2).

```python
def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        
        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}

"""
    
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    
    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""
        
    return prompt
```

```python
example_indices_full = [40]
example_index_to_summarize = 200

one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(one_shot_prompt)
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2049.png)

Now pass this prompt to perform the one shot inference:

```python
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2050.png)

### Few Shot Inference

Let's explore few shot inference by adding two more full dialogue-summary pairs to your prompt.

```python
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200

few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(few_shot_prompt)
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2051.png)

<aside>
💡 Other two shots (examples) not shown in the picture!

</aside>

Now pass this prompt to perform a few shot inference:

```python
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2052.png)

In this case, few shot did not provide much of an improvement over one shot inference. And, anything above 5 or 6 shot will typically not help much, either. Also, you need to make sure that you do not exceed the model's input-context length which, in our case, if 512 tokens. Anything above the context length will be ignored.

However, you can see that feeding in at least one full example (one shot) provides the model with more information and qualitatively improves the summary overall.

## Generative Configuration Parameters for Inference

You can change the configuration parameters of the `generate()` method to see a different output from the LLM. So far the only parameter that you have been setting was `max_new_tokens=50`, which defines the maximum number of tokens to generate. A full list of available parameters can be found in the [Hugging Face Generation documentation](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig).

A convenient way of organizing the configuration parameters is to use `GenerationConfig` class.

**Exercise:**

Change the configuration parameters to investigate their influence on the output.

Putting the parameter `do_sample = True`, you activate various decoding strategies which influence the next token from the probability distribution over the entire vocabulary. You can then adjust the outputs changing `temperature` and other parameters (such as `top_k` and `top_p`).

Uncomment the lines in the cell below and rerun the code. Try to analyze the results. You can read some comments below.

```python
generation_config = GenerationConfig(max_new_tokens=50)
# generation_config = GenerationConfig(max_new_tokens=10)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
```

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2053.png)

<aside>
💡 Comments related to the choice of the parameters in the code cell above:

- Choosing `max_new_tokens=10` will make the output text too short, so the dialogue summary will be cut.
- Putting `do_sample = True` and changing the temperature value you get more flexibility in the output.

</aside>

> As you can see, prompt engineering can take you a long way for this use case, but there are some limitations. Next, you will start to explore how you can use fine-tuning to help your LLM to understand a particular use case in better depth!
>

---

---

# Pre-training Large Language Models

- Decide between existing models or training from scratch.
- Training your own model from scratch has advantages in specific cases.
- Typically, start application development with an existing foundation model.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2054.png)

This step is called pre-training. LLMs learn language by studying huge amounts of text from the internet and curated sources. This helps them understand patterns for their training. Their "knowledge" is updated to improve their learning process. Pre-training needs a lot of computing power, usually GPUs.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2055.png)

<aside>
💡 When getting data from the internet, it's important to improve its quality and reduce bias. Usually, only a small part of the data is used for pre-training, affecting how much data you need.

</aside>

You observed three types of the transformer model: encoder-only, encoder-decoder, and decoder-only. Each is trained for a distinct purpose, enabling them to perform different tasks.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2056.png)

## Auto-encoding Models (Encoder only)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2057.png)

- They learn by predicting missing words in sentences (masked language modeling).
- This helps them understand the sentence's context from all sides. T
- They're great for tasks like classifying sentences (sentiment analysis) or working with individual words (named entity recognition).

- Encoder-only models are also called Autoencoding models.
- These models are pre-trained using masked language modeling.
- Masked tokens in input sequences are predicted to reconstruct original sentences.
- Denoising objective is achieved, enhancing representation learning.
- Encoder-only models capture bi-directional context for each token.
- BERT and RoBERTa are popular examples.

## Autoregressive Model (Decoder Only)

- The goal here is to guess the next word based on the words before it.
- This is sometimes called full language modeling.
- Decoder-based models predict the next word but only see the words leading up to it.
- They don't know when a sentence ends. These models go through each word one by one to predict what comes next.
- Unlike the encoder, this focuses on one direction. By practicing this prediction extensively, the model learns language patterns.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2058.png)

- These models use only the decoder part of the original design, excluding the encoder.
- Decoder-only models are often used for creating text, but larger ones can also handle various tasks even without training.
- GBT and BLOOM are examples of decoder-based models.

### Sequence to Sequence Modeling

The last transformer model variation is the sequence-to-sequence model, which combines both the encoder and decoder parts of the original transformer design. The specific pre-training objectives differ among models. A popular example is T5 (see below the HF doc), which pre-trains the encoder using span corruption. This involves masking random sequences of input tokens and replacing them with a unique Sentinel token (represented as x). Sentinel tokens are special and don't correspond to actual words from the text. The decoder's task is then to reconstruct the masked token sequences sequentially. The output starts with the Sentinel token, followed by the predicted tokens.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2059.png)

Sequence-to-sequence models like T5 are used for translation, summarization, and question-answering. They are valuable when you have both input and output texts. Alongside T5, another renowned encoder-decoder model is BART.

[T5](https://huggingface.co/docs/transformers/model_doc/t5)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2060.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2061.png)

# Computational Challengers of training LLMs

When working with large language models, a common problem is running out of memory.  CUDA is a set of tools for Nvidia GPUs, used by frameworks like PyTorch and TensorFlow to speed up tasks like matrix calculations in deep learning.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2062.png)

If you've used Nvidia GPUs for training or loading models, you might have seen this error message.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2063.png)

 A 32-bit float occupies four bytes of memory.

A single parameter is usually shown as a 32-bit float, which is a computer's method of representing real numbers. To store a billion parameters, you'd require four gigabytes of GPU RAM at 32-bit full precision. This amounts to a significant memory usage.

When considering model training, you must factor in extra components that consume GPU memory. These encompass two Adam optimizer states, gradients, activations, and temporary variables essential for functions. This often results in an additional 20 bytes of memory per model parameter.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2064.png)

To train a one billion parameter model at 32-bit full precision, you'll need approximately 80 gigabyte of GPU RAM

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2065.png)

## Quantization

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2066.png)

This technique helps save memory.

A method to decrease memory usage is quantization. The concept involves reducing weight precision from 32-bit floating point to 16-bit floating point or eight-bit integers. In deep learning frameworks, corresponding data types are FP32 for 32-bit, FP16 or Bfloat16 for 16-bit half precision, and int8 for eight-bit integers.

### FP16

To represent PI with six decimals, you use floating point numbers stored as bits. FP32 uses 32 bits: 1 sign bit, 8 exponent bits, and 23 fraction bits. When converting back, you lose some precision.

Now, projecting PI from FP32 to FP16, which has 16 bits, you get 3.140625 with six decimal places. This slight loss is usually okay since it optimizes memory. FP32 needs 4 bytes, FP16 only 2 bytes.

### BFLOAT16 | BF16

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2067.png)

Quantization cuts memory usage in half.

One specific data type gaining popularity is BFLOAT16 (BF16), developed at Google Brain. It's becoming a favored choice in deep learning, with LLMs like FLAN-T5 being pre-trained using BF16. BF16 is a hybrid between FP16 and FP32, offering stability during training. Newer GPUs like NVIDIA's A100 support BF16.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2068.png)

BF16 captures the dynamic range of a full 32-bit float but uses only 16 bits. It allocates all eight bits for the exponent and truncates the fraction to seven bits. This enhances performance by saving memory and accelerating calculations. The trade-off is that BF16 isn't ideal for integer operations, which are uncommon in deep learning.

### INT8

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2069.png)

This reduces memory needs from four bytes to just one, but sacrifices precision.

For a comprehensive view, consider quantizing Pi from 32-bit to even lower 8-bit precision. Allocating one bit for the sign, INT8 values use the remaining seven bits. This allows representation of numbers from -128 to +127. As a result, Pi becomes approximately 2 or 3 in the 8-bit space.

### Summary

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2070.png)

**Storage needed to load 1 Billion Parameters**

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2071.png)

Approximate GPU ram needed to train 1 Billion parameters

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2072.png)

Comparison (I am out of headings ;((((((( )

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2073.png)

# BloombergGPT

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2074.png)

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2075.png)

BloombergGPT, developed by Bloomberg, is a large Decoder-only language model. It underwent pre-training using an extensive financial dataset comprising news articles, reports, and market data, to increase its understanding of finance and enabling it to generate finance-related natural language text. The datasets are shown in the image above.

![Untitled](Week%201%20Transformers,%20Project%20Life%20cycle%20and%20Hyper%20%20e8be168705244c2f9028eb6a2043d925/Untitled%2076.png)

During the training of BloombergGPT, the authors used the Chinchilla Scaling Laws to guide the number of parameters in the model and the volume of training data, measured in tokens. The recommendations of Chinchilla are represented by the lines Chinchilla-1, Chinchilla-2 and Chinchilla-3 in the image, and we can see that BloombergGPT is close to it.

While the recommended configuration for the team’s available training compute budget was 50 billion parameters and 1.4 trillion tokens, acquiring 1.4 trillion tokens of training data in the finance domain proved challenging. Consequently, they constructed a dataset containing just 700 billion tokens, less than the compute-optimal value. Furthermore, due to early stopping, the training process terminated after processing 569 billion tokens.

The BloombergGPT project is a good illustration of pre-training a model for increased domain-specificity, and the challenges that may force trade-offs against compute-optimal model and training configurations.

You can read the BloombergGPT article [here](https://arxiv.org/abs/2303.17564).

# Quiz

Question 1
Interacting with Large Language Models (LLMs) differs from traditional machine learning models.  Working with LLMs involves natural language input, known as a  _____, resulting in output from the Large Language Model, known as the ______ .

Choose the answer that correctly fill in the blanks.

→ prompt, completion

tunable request, completion

prompt, fine-tuned LLM

prediction request, prediction response

Question 2
Large Language Models (LLMs) are capable of performing multiple tasks supporting a variety of use cases.  Which of the following tasks supports the use case of converting code comments into executable code?

Text summarization

Information Retrieval

Invoke actions from text

→ Translation

Question 3
What is the self-attention that powers the transformer architecture?

A measure of how well a model can understand and generate human-like language.

A technique used to improve the generalization capabilities of a model by training it on diverse datasets.

→ A mechanism that allows a model to focus on different parts of the input sequence during computation.

The ability of the transformer to analyze its own performance and make adjustments accordingly.

Question 4
Which of the following stages are part of the generative AI model lifecycle mentioned in the course? (Select all that apply)

Performing regularization

→ Defining the problem and identifying relevant datasets.

→ Deploying the model into the infrastructure and integrating it with the application.

→ Manipulating the model to align with specific project needs.

→ Selecting a candidate model and potentially pre-training a custom model.

Question 5
"RNNs are better than Transformers for generative AI Tasks."

Is this true or false?

True
→ False

Question 6
Which transformer-based model architecture has the objective of guessing a masked token based on the previous sequence of tokens by building bidirectional representations of the input sequence.

→ Autoencoder

Autoregressive

Sequence-to-sequence

Question 7
Which transformer-based model architecture is well-suited to the task of text translation?

Autoencoder
→ Sequence-to-sequence
Autoregressive

Question 8
Do we always need to increase the model size to improve its performance?
True

→ False

Question 9
Scaling laws for pre-training large language models consider several aspects to maximize performance of a model within a set of constraints and available scaling choices.  Select all alternatives that should be considered for scaling when performing model pre-training?

1 point

→Compute budget: Compute constraints

→ Model size: Number of parameters

→ Dataset size: Number of tokens

Batch size: Number of samples per iteration

Question 10
"You can combine data parallelism with model parallelism to train LLMs."

Is this true or false?

→ True
False

## Reading Resources

Below you'll find links to the research papers discussed in this weeks videos. You don't need to understand all the technical details discussed in these papers - **you have already seen the most important points you'll need to answer the quizzes** in the lecture videos.

However, if you'd like to take a closer look at the original research, you can read the papers and articles via the links below.

### **Transformer Architecture**

- **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)** - This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.
- **[BLOOM: BigScience 176B Model](https://arxiv.org/abs/2211.05100)**  - BLOOM is a open-source LLM with 176B parameters (similar to GPT-4) trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model. You can also see a high-level overview of the model [here](https://www.notion.so/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4?pvs=21).
- **[Vector Space Models](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)** - Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.

### **Pre-training and scaling laws**

- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** empirical study by researchers at OpenAI exploring the scaling laws for large language models.

### **Model architectures and pre-training objectives**

- **[What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)** - The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.
- **[HuggingFace Tasks](https://huggingface.co/tasks) and [Model Hub](https://huggingface.co/models)** - Collection of resources to tackle varying machine learning tasks using the HuggingFace library.
- **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)** - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)

### **Scaling laws and compute-optimal models**

- **[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)**  ****This paper investigates the potential of few-shot learning in Large Language Models.
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)** Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.
- **[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)** - LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.

# **Acknowledgements**

Screenshots are from the course ***Generative AI and LLMs*** from [deeplearning.ai](https://deeplearning.ai) on [coursera](https://coursera.org) and texts are summarized using ChatGPT and BARD ai
