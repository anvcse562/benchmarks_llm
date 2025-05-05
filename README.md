# benchmarks_llm

# Language Model Benchmarks

This document outlines code snippets and usage examples for evaluating language models on various benchmarks, including zero-shot and few-shot tasks.

## 1. MMLU (Massive Multitask Language Understanding)

MMLU evaluates a model's performance across various tasks, including elementary mathematics, US history, and more.

```python
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from datasets import load_dataset

model = AutoModelForMultipleChoice.from_pretrained("facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

dataset = load_dataset("mmlu", "en")
inputs = tokenizer(dataset['train']['question'], padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
```

## 2. BBH (Big-Bench Hard)


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-xl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

dataset = load_dataset("bigbench", "hard")
inputs = tokenizer(dataset['train']['input'], padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
```

## 3. DROP (Discrete Reasoning Over Paragraphs)


```python
from drop import DROP
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
drop = DROP(model=model, tokenizer=tokenizer)

results = drop.evaluate()
print("DROP Accuracy:", results['accuracy'])
```


## 4. AGIEval (Artificial General Intelligence Evaluation)


```python
from agieval import AGIEval
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

agieval = AGIEval(model=model, tokenizer=tokenizer)
results = agieval.evaluate()
print("AGIEval Accuracy:", results['accuracy'])
```


##  5. GSM8K (Grade School Math 8K)


```python
from gsm8k import GSM8K
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

gsm8k = GSM8K(model=model, tokenizer=tokenizer)
results = gsm8k.evaluate()
print("GSM8K Accuracy:", results['accuracy'])
```


## 6. MATH (Challenging High School-Level Math)


```python
from math_bench import MATH
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

math_bench = MATH(model=model, tokenizer=tokenizer)
results = math_bench.evaluate()
print("MATH Accuracy:", results['accuracy'])
```


## 7. MGSM (Multilingual Grade School Math)


```python
from mgsm import MGSM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

mgsm = MGSM(model=model, tokenizer=tokenizer)
results = mgsm.evaluate()
print("MGSM Accuracy:", results['accuracy'])
```


## 8. HumanEval


```python
from human_eval import HumanEval
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

human_eval = HumanEval(model=model, tokenizer=tokenizer)
results = human_eval.evaluate()
print("HumanEval Accuracy:", results['accuracy'])
```

## 9. MBPP (Mostly Basic Python Programming)


```python
from mbpp import MBPP
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

mbpp = MBPP(model=model, tokenizer=tokenizer)
results = mbpp.evaluate()
print("MBPP Accuracy:", results['accuracy'])
```


## 10. LiveCodeBench (Interactive Code Generation)


```python
from livecodebench import LiveCodeBench
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

livecodebench = LiveCodeBench(model=model, tokenizer=tokenizer)
results = livecodebench.evaluate()
print("LiveCodeBench Accuracy:", results['accuracy'])
```



## 11. CMMLU (Chinese MMLU)


```python
from cmmlu import CMMLU
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

cmmlu = CMMLU(model=model, tokenizer=tokenizer)
results = cmmlu.evaluate()
print("CMMLU Accuracy:", results['accuracy'])
```


## benchmark illustration


```python

from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "DeepSeek-v2", "Qwen-2.5", "LLaMA-3.1", "DeepSeek-v3",
    "GPT-4", "Claude-Opus", "Gemini-1.5-Pro", "Claude-Sonnet"
]

def run_benchmark(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    mmlu_results = MMLU(model=model, tokenizer=tokenizer).evaluate()
    bbh_results = BBH(model=model, tokenizer=tokenizer).evaluate()
    drop_results = DROP(model=model, tokenizer=tokenizer).evaluate()
    
    print(f"Model: {model_name}")
    print(f"MMLU Accuracy: {mmlu_results['accuracy']}")
    print(f"BBH Accuracy: {bbh_results['accuracy']}")
    print(f"DROP Accuracy: {drop_results['accuracy']}")
    print("-" * 50)

for model in models:
    run_benchmark(model)

```


## Sample Task Views

### MMLU Sample

| Task            | Question                                     | Options                                   |
|-----------------|----------------------------------------------|-------------------------------------------|
| Elementary Math | What is 5 + 3?                                | A) 7, B) 8, C) 9, D) 10                    |
| US History      | Who was the first president of the U.S.?     | A) George Washington, B) Lincoln, etc.    |

---

### BBH Sample

| Task       | Input                            | Expected Output |
|------------|----------------------------------|-----------------|
| Arithmetic | What is 12 times 15?             | 180             |
| Analogies  | Man is to woman as king to ___?  | Queen           |

---

### MATH Sample

| Problem   | Input                          | Solution |
|-----------|--------------------------------|----------|
| Algebra   | Solve for x: 2x + 3 = 7         | x = 2    |
| Geometry  | Area of circle, radius = 5      | 78.54    |

Note: Most of these benchmarks support zero-shot or few-shot evaluation without fine-tuning the model.

