# ***Fine Tuning***

---

- [***Fine Tuning***](#fine-tuning)
- [1. Instruction Fine Tuning](#1-instruction-fine-tuning)
  - [1.1 What is the need?](#11-what-is-the-need)
  - [1.2 What is Fine Tuning](#12-what-is-fine-tuning)
  - [1.2 But Dataset?](#12-but-dataset)
    - [1.2.1 Divide The Dataset](#121-divide-the-dataset)
  - [1.3 Working with data](#13-working-with-data)
- [2. Fine-Tuning on Single Tasks](#2-fine-tuning-on-single-tasks)
  - [2.1 Catastrophic Forgetting](#21-catastrophic-forgetting)
  - [2.2 Avoiding Catastrophic Forgetting](#22-avoiding-catastrophic-forgetting)
- [3. Fine Tuning Multi-tasks](#3-fine-tuning-multi-tasks)
  - [3.1 FLAN Models](#31-flan-models)
    - [3.1.1 Flan T5 Models](#311-flan-t5-models)
  - [3.2 Fine Tuning in Customer Service Chatbot](#32-fine-tuning-in-customer-service-chatbot)
    - [3.2.1 Before FineTuning](#321-before-finetuning)
    - [3.2.2 After Fine-Tuning](#322-after-fine-tuning)
  - [3.3 Read the Paper](#33-read-the-paper)
- [4. Model/LLM Evaluation Challenges](#4-modelllm-evaluation-challenges)
  - [4.1 The metrics: Rouge and Bleu](#41-the-metrics-rouge-and-bleu)
    - [4.1.1 ?-grams](#411--grams)
  - [4.2 ROUGE](#42-rouge)
    - [4.2.1 ROUGE-1](#421-rouge-1)
    - [4.2.2 ROUGE-2](#422-rouge-2)
    - [4.2.3 ROUGE-L](#423-rouge-l)
  - [4.3 BLEU Score](#43-bleu-score)
- [5. Parameter Efficient fine-tuning (PEFT)](#5-parameter-efficient-fine-tuning-peft)
  - [5.1 Recall: Challenges of Full Fine Tuning](#51-recall-challenges-of-full-fine-tuning)
  - [5.2 PEFT](#52-peft)
    - [Learn More about PEFT](#learn-more-about-peft)
- [6. PEFT Techniques](#6-peft-techniques)
    - [6.0.0 Transformers: Recap](#600-transformers-recap)
  - [6.1 LoRA](#61-lora)
    - [6.1.1 How does this lower the parameters? Example](#611-how-does-this-lower-the-parameters-example)
    - [6.1.2 ROUGE Metrics for Full vs LoRA Fine-tuning](#612-rouge-metrics-for-full-vs-lora-fine-tuning)
  - [6.2 Soft Prompts and Prompt Tuning](#62-soft-prompts-and-prompt-tuning)
    - [6.2.1 Prompt Tuning is Not Prompts Engineering](#621-prompt-tuning-is-not-prompts-engineering)
    - [6.2.2 Prompt Tuning](#622-prompt-tuning)
    - [6.2.3 Full vs. Prompt Tuning](#623-full-vs-prompt-tuning)
    - [6.2.4 Prompt Tuning for Multi-Tasks](#624-prompt-tuning-for-multi-tasks)
    - [6.2.5 Performance Comparison](#625-performance-comparison)
- [7. Resources](#7-resources)
  - [**Multi-task, instruction fine-tuning**](#multi-task-instruction-fine-tuning)
  - [**Model Evaluation Metrics**](#model-evaluation-metrics)
  - [**Parameter- efficient fine tuning (PEFT)**](#parameter--efficient-fine-tuning-peft)
  - [**LoRA**](#lora)
  - [**Prompt tuning with soft prompts**](#prompt-tuning-with-soft-prompts)
- [**Acknowledgements**](#acknowledgements)

# 1. Instruction Fine Tuning

## 1.1 What is the need?

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled.png)

The week covers methods to enhance the performance of an existing model for a specific use case. It also delves into important metrics for evaluating the fine-tuned LLM's performance and quantifying its improvement over the initial base model.

Let's start by discussing how to fine tune an LLM with instruction prompts.

Last week revealed that specific models have the capacity to discern instructions within a prompt, leading to precise zero-shot inference.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%201.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%202.png)

Yet, this method has a couple of downsides. First, it doesn't always work for smaller models, even with five or six examples. Second, any examples in your prompt use up space in the context window, reducing room for other important information.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%203.png)

On the other hand, as evidenced by the given example, smaller LLMs may struggle.

Furthermore, you gain insight into the fact that presenting one or more task examples, termed as one-shot or few-shot inference, can assist the model in recognizing the task and generating an appropriate completion.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%204.png)

## 1.2 What is Fine Tuning

You can use a method called fine-tuning to improve a base model further. Unlike pre-training, which involves training an LLM using lots of text, fine-tuning is more like supervised learning. It uses labeled examples to adjust the model's weights. These examples are pairs of prompts and their completions. This fine-tuning process helps the model get better at generating completions for a specific task.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%205.png)

One useful strategy is called instruction fine-tuning, which can improve the model's performance on different tasks.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%206.png)

Instruction fine-tuning involves training the model with examples that show how it should react to a particular instruction. Here are a few prompt examples to illustrate this concept. In both cases, the instruction is "classify this review," and the expected completion is a text string that begins with "sentiment" followed by either "positive" or "negative.”

Your training dataset contains prompt completion pairs for your specific task, each with an instruction. For instance, to improve summarization, you'd use examples starting with "summarize."

These examples help the model generate responses following the given instructions.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%207.png)

For translation, instructions like "translate this sentence" would be included.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%208.png)

Utilizing memory optimization and parallel computing strategies, as covered last week, can prove advantageous in this context.

Instruction fine-tuning, involving updates to all model weights, is called full fine-tuning. This process produces an updated model version with new weights.

Similar to pre-training, full fine-tuning demands sufficient memory and computational resources for storing and processing gradients, optimizers, and other training components.

## 1.2 But Dataset?

First, get your training data ready. There are datasets used for training earlier language models, but they may not have instructions. Luckily, there are prompt template libraries that can help. These templates can turn existing datasets, like Amazon product reviews, into instruction prompts for fine-tuning.

These libraries have templates for different tasks and datasets.

Here are three prompts designed for the Amazon reviews dataset, suitable for fine-tuning models in classification, text generation, and text summarization tasks. In each case, the original review (referred to as review_body) is fed into the template. The template starts with an instruction like "predict the associated rating," "generate a star review," or "give a short sentence describing the following product review."

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%209.png)

This creates a prompt that combines the instruction with an example from the dataset.

### 1.2.1 Divide The Dataset

With your instruction dataset ready, divide it into training, validation, and test sets. During fine-tuning, pick prompts from the training data and input them into the LLM. Compare the LLM's generated completions with expected responses from the data. In this case, the model's classification of the review as neutral seems a bit understated.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2010.png)

## 1.3 Working with data

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2011.png)

This sequence occurs across multiple batches of prompt completions over several epochs, gradually refining the model's task performance.

Remember, an LLM's output forms a probability distribution across tokens. To assess, compare the distribution of completions with the training labels using standard cross-entropy – a way to measure the difference between these token distributions. This cross-entropy-driven loss guides backpropagation, a method to adjust model weights based on errors.

Just like regular supervised learning, define evaluation steps for LLM performance using validation data to measure validation accuracy. Post fine-tuning, assess final performance using a separate test dataset to determine test accuracy. This refines the base model into an improved version, better suited for your specific tasks.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2012.png)

# 2. Fine-Tuning on Single Tasks

Although LLMs are known for their diverse language abilities, your application might require just one specific task. In such cases, you can fine-tune a pre-trained model solely to enhance its performance on that particular task of interest.

For example, summarization using a dataset of examples for that task. Interestingly, good results can be achieved with relatively few examples. Often just 500-1,000 examples can result in good performance in contrast to the billions of pieces of texts that the model saw during pre-training. However, there is a potential downside to fine-tuning on a single task

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2013.png)

## 2.1 Catastrophic Forgetting

This process can result in a phenomenon known as catastrophic forgetting. This occurs because full fine-tuning alters the original LLM's weights. While it improves performance on the specific fine-tuning task, it can negatively impact performance on other tasks.

For instance, fine-tuning can enhance a model's sentiment analysis capability for a review, yielding quality results. However, this focus may cause the model to forget other tasks.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2014.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2015.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2016.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2017.png)

Before fine-tuning, the model accurately recognized named entities, like identifying "Charlie" as a cat's name in a sentence. Yet, after fine-tuning, the model might struggle with this task, getting both the entity identification wrong and displaying behavior associated with the new task.

## 2.2 Avoiding Catastrophic Forgetting

How can you avoid catastrophic forgetting? First, determine if it impacts your specific case. If you only need strong performance on the fine-tuned task, neglecting other tasks might not be an issue. To maintain multitasking capabilities, you can perform concurrent fine-tuning on multiple tasks, albeit requiring more data and computation. Another option is parameter-efficient fine-tuning (PEFT), an alternative to full fine-tuning.

# 3. Fine Tuning Multi-tasks

Multitask fine-tuning extends single task fine-tuning by training on a dataset with examples for various tasks. This mixed dataset guides the model to improve performance across tasks like summarization, rating, translation, and recognition simultaneously, avoiding catastrophic forgetting. With multiple training epochs, losses update the model's weights, resulting in an instruction-tuned model skilled in multiple tasks.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2018.png)

A drawback is the need for substantial data – around 50-100,000 examples. Despite this, the effort yields capable models suitable for scenarios requiring strong performance across tasks.

## 3.1 FLAN Models

Let's explore a group of models trained using multitask instruction fine-tuning. Instruct model variations arise from the datasets and tasks employed in fine-tuning. One instance is the FLAN family of models, short for fine-tuned language net. FLAN fine-tuning marks the last training step, earning it the metaphorical title of "dessert" to the "main course" of pre-training – quite fitting. Examples include FLAN-T5, the FLAN instruct version of the T5 foundation model, and FLAN-PALM, the instruct version of the palm foundation model.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2019.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2020.png)

### 3.1.1 Flan T5 Models

FLAN-T5 is a great general purpose instruct model. In total, it's been fine tuned on 473 datasets across 146 task categories. Those datasets are chosen from other models and papers as shown here.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2021.png)

FLAN-T5 underwent fine-tuning on 473 datasets across 146 tasks, drawn from different models and papers.

An example is SAMSum, part of the muffin collection, used for dialogue summarization.

SAMSum contains 16,000 messenger-like conversations with summaries, as shown here.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2022.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2023.png)

Presented is a prompt template tailored for the SAMSum dialogue summary dataset. This template consists of various instructions, all essentially requesting the model to achieve a single task: summarizing a dialogue. Variations include "Briefly summarize that dialogue," "Provide a summary of this dialogue," and "Explain what was happening in that conversation." Offering diverse phrasings aids the model's generalization and enhances performance.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2024.png)

## 3.2 Fine Tuning in Customer Service Chatbot

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2025.png)

FLAN-T5 excels in various tasks, but you might seek improvements for your specific use. For instance, as a data scientist, developing a chatbot-backed app for customer support, summaries of dialogues are crucial.

While SAMSum aids FLAN-T5 in summarization, its examples differ from the language structure of customer service chats, necessitating further refinement.

You can further refine FLAN-T5 by fine-tuning it with a dialogue dataset that closely matches your chatbot interactions. This is precisely what you'll explore in this week's lab. Using a domain-specific summarization dataset called dialogsum, containing over 13,000 support chat dialogues and summaries, you aim to improve FLAN-T5's ability to summarize customer service chats. Notably, the dialogsum dataset is new to FLAN-T5, offering fresh conversation contexts for better performance.

![Dialog Specific Dataset: DialogSUM](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2026.png)

Dialog Specific Dataset: DialogSUM

### 3.2.1 Before FineTuning

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2027.png)

Let's see FLAN-T5's initial response to the prompt, before further fine-tuning. Note the condensed prompt on the left for better visibility of the model's completion. The model identifies the topic as a reservation for Tommy, but it lags behind the human-generated baseline summary. The baseline captures more details, like Mike's check-in query, while the model invents extra information, like the hotel name and city.

### 3.2.2 After Fine-Tuning

Now, let's see how the model performs after fine-tuning on the dialogsum dataset. You'll notice it's closer to a human-generated summary without fabricated details. In real-world applications, utilizing your company's own data, like support chat conversations, enhances the model's understanding of your preferred summarization style, benefiting your customer service team.

## 3.3 Read the Paper

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2028.png)

This example uses dialogsum to illustrate custom data fine-tuning.

[This paper](https://arxiv.org/abs/2210.11416) introduces FLAN (Fine-tuned LAnguage Net), an instruction finetuning method, and presents the results of its application. The study demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating Chain-of-Thought Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. The paper also provides detailed information on how each these aspects was evaluated.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2029.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2030.png)

# 4. Model/LLM Evaluation Challenges

In traditional machine learning, you measure a model's performance by checking how well it does on known training and validation data. Simple metrics like accuracy, which shows the fraction of correct predictions, are used because these models are predictable.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2031.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2032.png)

While humans can grasp these differences, training models on vast data demands an automated approach.

Large language models pose evaluation challenges due to non-deterministic outputs and language complexities. Comparing sentences like "Mike really loves drinking tea" and "Mike adores sipping tea" is tough. Even slight phrasing changes, as in "Mike does not drink coffee" and "Mike does drink coffee," can entirely alter meaning.

## 4.1 The metrics: Rouge and Bleu

ROUGE and BLEU are widely used evaluation metrics for various tasks. ROUGE assesses the quality of automatically generated summaries by comparing them to human reference summaries. BLEU evaluates the quality of machine-translated text by comparing it to human translations.

### 4.1.1 ?-grams

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2033.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2034.png)

Before delving into metric calculations, it's important to understand some language terminology. In the language structure, a unigram refers to a single word, a bigram consists of two words, and an n-gram is a group of n words.

## 4.2 ROUGE

### 4.2.1 ROUGE-1

Let's begin by examining the ROUGE-1 metric. Consider a human-generated reference sentence and a corresponding generated output. Metric calculations involve recall, precision, and F1, similar to other machine-learning tasks. Recall assesses matched words or unigrams between reference and generated output, yielding a perfect score (1) when all words match. Precision calculates unigram matches divided by output size, while F1 combines these values harmonically.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2035.png)

These metrics focus solely on individual words and disregard word order, which can be misleading. High scores might indicate well-performing sentences, but subjective quality may differ.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2036.png)

Consider a scenario where the model's generated sentence differs by just one word, such as "Not, so it is not cold outside." Despite the change, the scores would remain unchanged.

### 4.2.2 ROUGE-2

To improve scores, you can look at pairs of words from the reference and generated sentence. This helps recognize how words are ordered in the sentence. This approach is called using "bigrams," and it allows you to calculate a metric called ROUGE-2.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2037.png)

Working with word pairs shows sentence word order in a simple way. Using bigrams helps calculate ROUGE-2. Calculate recall, precision, and F1 with bigrams, not single words. These scores might be lower than ROUGE-1, especially in longer sentences where bigrams can mismatch more, leading to even lower scores.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2038.png)

### 4.2.3 ROUGE-L

Instead of progressing with larger ROUGE numbers for n-grams of three or four, let's explore a different method.

You'll focus on finding the longest shared sequence between the generated and reference outputs.

In this instance, the longest matching sub-sequences are "it is" and "cold outside," each with a length of two.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2039.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2040.png)

Now, you can use the LCS value for recall, precision, and F1 calculation. The longest common subsequence length (here, two) is the numerator in recall and precision. These form the Rouge-L score. Keep in mind, Rouge scores are meaningful only when comparing models for the same task, like summarization.

## 4.3 BLEU Score

Another valuable performance metric is the BLEU score, short for bilingual evaluation understudy.

Remember, the BLEU score assesses the quality of machine-translated text. It's computed by averaging precision across various n-gram sizes, similar to the Rouge-1 score we discussed earlier.

However, BLEU is calculated for a range of n-gram sizes and then averaged.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2041.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2042.png)

While Rouge calculations were detailed earlier, BLEU results can be obtained using convenient libraries like Hugging Face's. I've utilized these libraries to assess our candidate sentences.

BLEU score assesses translation quality by comparing n-grams in machine and reference translations. Calculate it by averaging precision across various n-gram sizes. To compute the BLEU score manually, perform multiple calculations and average the results. For a clearer understanding, let's consider a longer sentence. The human reference sentence is "I am very happy to say that I am drinking a warm cup of tea."

The initial candidate is "I am very happy that I am drinking a cup of tea." The BLEU score is 0.495. As the sentence aligns more closely with the original, the score approaches one. Both Rouge and BLEU are straightforward and inexpensive metrics to calculate. They serve as handy references during model iteration.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2043.png)

However, for a comprehensive evaluation of a large language model, don't rely solely on these metrics.

<aside>
💡 Instead, use Rouge for summarization tasks and BLEU for translation tasks, mainly for diagnostic assessment.

</aside>

# 5. Parameter Efficient fine-tuning (PEFT)

## 5.1 Recall: Challenges of Full Fine Tuning

Training LLMs requires significant computing power. Full fine-tuning needs memory for model, optimizer, gradients, activations, and more. Even if your computer fits the model (often hundreds of gigabytes), other components can surpass it, straining consumer hardware.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2044.png)

## 5.2 PEFT

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2045.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2046.png)

- Contrasting full fine-tuning, where all model weights update during supervised learning, parameter-efficient fine-tuning (PEFT) alters only a small subset.
- Certain techniques freeze most weights, focusing on specific layers or components.
- Others introduce new parameters or layers, fine-tuning only those.
- With PEFT, most LLM weights remain frozen, significantly reducing trained parameters compared to the original model, often just 15-20%.
- This greatly eases memory demands during training.

PEFT is often achievable on a single GPU. Moreover, as the original LLM is minimally altered or remains intact, PEFT is less susceptible to the memory-related challenges of full fine-tuning. Unlike full fine-tuning, which generates a new model version per task, each being as large as the original, PEFT proves more storage-friendly when *fine-tuning for multiple tasks.*

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2047.png)

Parameter-efficient fine-tuning trains only a few weights, leading to a much smaller overall size, potentially as compact as megabytes, task-dependent. These new parameters work alongside the original LLM weights during inference. PEFT's task-specific weights are trainable and can be seamlessly replaced for inference, enabling the original model to efficiently adapt to various tasks.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2048.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2049.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2050.png)

### Learn More about PEFT

[Fine-tuning LLMs with PEFT and LoRA](https://youtu.be/Us5ZFp16PaU)

# 6. PEFT Techniques

### 6.0.0 Transformers: Recap

Here's the transformer architecture diagram you previously encountered. The input prompt becomes tokens, transformed into embedding vectors, and then fed into the encoder and/or decoder. In both these parts, two types of neural networks exist: self-attention and feedforward networks. These network weights are acquired during pre-training.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2051.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2052.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2053.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2054.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2055.png)

## 6.1 LoRA

LoRA is a method that reduces fine-tuning parameter count. It freezes original model parameters and introduces rank decomposition matrices. Smaller matrices' dimensions are chosen to yield the same size as the weights they modify. The original LLM weights stay frozen, while smaller matrices undergo supervised learning, akin to what you've seen.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2056.png)

During inference, these matrices multiply to form a dimension-matching matrix, added to the original weights. This updated model, with the same parameter count as the original, performs your task. This minimally affects inference speed.

### 6.1.1 How does this lower the parameters? Example

Researchers have discovered that applying LoRA to the self-attention layers of the model often suffices for fine-tuning and performance enhancement. While it's possible to use LoRA on other parts like feed-forward layers, most LLM parameters reside in attention layers. This leads to significant parameter reduction by applying LoRA to these weight matrices.

Consider a practical example using the transformer architecture from the "Attention is All You Need" paper. The paper states transformer weights as 512 by 64 dimensions, resulting in 32,768 trainable parameters per weights matrix. When using LoRA with a rank of eight for fine-tuning, you train two smaller rank decomposition matrices with a dimension of eight. This results in 512 parameters for Matrix A (8 by 64) and 4,096 parameters for Matrix B (512 by 8). By updating these new low-rank matrices instead of original weights, you train 4,608 parameters instead of 32,768—a reduction of 86%.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2057.png)

<aside>
💡 LoRA's parameter reduction lets you efficiently fine-tune with a single GPU, skipping the need for a GPU cluster.

</aside>

### 6.1.2 ROUGE Metrics for Full vs LoRA Fine-tuning

Let's compare ROUGE scores for three models: the FLAN-T5 base, fully fine-tuned, and LoRA fine-tuned. Focusing on dialogue summarization with FLAN-T5, baseline scores are low. Full fine-tuning raises ROUGE 1 by 0.19, significantly improving performance. LoRA fine-tuning also boosts ROUGE 1 by 0.17, using fewer resources. This efficiency trade-off could be worthwhile.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2058.png)

LoRA aims to update model weights efficiently, avoiding complete retraining of each parameter. PEFT includes additive methods too, enhancing performance without altering weights.

## 6.2 Soft Prompts and Prompt Tuning

### 6.2.1 Prompt Tuning is Not Prompts Engineering

Prompt tuning and prompt engineering are distinct. The latter involves modifying prompt language for desired completions, from word changes to examples. This aids the model's understanding but has drawbacks: manual effort, context window limits, and uncertain task performance.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2059.png)

### 6.2.2 Prompt Tuning

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2060.png)

Prompt tuning involves adding trainable tokens to your prompt, allowing supervised learning to optimize their values. These trainable tokens form a "soft prompt," added before input text embedding vectors. Soft prompt vectors match language token embedding lengths, and 20 to 100 virtual tokens can often yield effective results.

Soft prompt vectors match the length of language token embedding vectors. Adding around 20 to 100 virtual tokens often leads to effective performance. In contrast, natural language tokens are hard, each corresponding to a fixed spot in the embedding vector space.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2061.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2062.png)

Soft prompts aren't rigid discrete words like in natural language.

They're flexible virtual tokens existing in a continuous, multi-dimensional embedding space.

With supervised learning, the model learns optimal values for these virtual tokens to enhance task performance.

### 6.2.3 Full vs. Prompt Tuning

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2063.png)

Full fine-tuning involves updating large language model weights using input prompts and output completions. In prompt tuning, the model's weights are frozen, and soft prompt embedding vectors are optimized for prompt completions. Prompt tuning trains only a few parameters, making it highly efficient compared to full fine-tuning, which involves training millions to billions of parameters, similar to approaches like LoRA

### 6.2.4 Prompt Tuning for Multi-Tasks

Train separate soft prompts for tasks, easily switch them for inference.

Compact on disk, efficient and versatile. Same LLM for all tasks, just change soft prompt at inference.

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2064.png)

### 6.2.5 Performance Comparison

Prompt tuning's performance varies based on model size. In comparison to full fine tuning (red) and multitask fine tuning (orange), prompt tuning (green) doesn't excel for smaller models. However, its performance improves as model size grows. Around 10 billion parameters, prompt tuning matches full fine tuning's effectiveness and outperforms prompt engineering (blue).

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2065.png)

![Untitled](Week%202%20Fine%20Tuning%2090b3956215bb4e6fa757bd96bd2a45d8/Untitled%2066.png)

# 7. Resources

Below you'll find links to the research papers discussed in this weeks videos. You don't need to understand all the technical details discussed in these papers - **you have already seen the most important points you'll need to answer the quizzes** in the lecture videos.

However, if you'd like to take a closer look at the original research, you can read the papers and articles via the links below.

## **Multi-task, instruction fine-tuning**

- **[Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)** - Scaling fine-tuning with a focus on task, model size and chain-of-thought data.
- **[Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)** - This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.

## **Model Evaluation Metrics**

- **[HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)** - HELM is a living benchmark to evaluate Language Models more transparently.
- **[General Language Understanding Evaluation (GLUE) benchmark](https://openreview.net/pdf?id=rJ4km2R5t7)** - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.
- **[SuperGLUE](https://super.gluebenchmark.com/)** - This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.
- **[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)** - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.
- **[Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/pdf/2009.03300.pdf)** - This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.
- **[BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models](https://arxiv.org/pdf/2206.04615.pdf)** - The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.

## **Parameter- efficient fine tuning (PEFT)**

- **[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)** - This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.
- **[On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583.pdf)** - The paper analyzes sparse fine-tuning methods for pre-trained models in NLP.

## **LoRA**

- **[LoRA Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)**  This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)** - This paper introduces an efficient method for fine-tuning large language models on a single GPU, based on quantization, achieving impressive results on benchmark tests.

## **Prompt tuning with soft prompts**

- **[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)** - The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.

# **Acknowledgements**

Screenshots are from the course ***Generative AI and LLMs*** from [deeplearning.ai](https://deeplearning.ai) on [coursera](https://coursera.org) and texts are summarized using ChatGPT and BARD ai
