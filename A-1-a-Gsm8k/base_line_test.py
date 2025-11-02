# In[ ]:
# paper: Gemma 2: Improving Open Language Models at a Practical Size
# Table 13.

# GSM8K(Benchmark) 5-shot(metric) 68.6 (Gemma-2 9B)

# In[ ]:
# pip install transformers torch accelerate datasets tqdm

# In[ ]:
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import os
from datetime import datetime


# In[ ]:
local_model_path = "/home/peternicholson/Documents/gemma-2-9b-it"
local_Gsm8k_path = "/home/peternicholson/Documents/Gsm8k-dataset/grade-school-math-master/grade_school_math/data/"
general_path = "/home/peternicholson/Documents/base-Gsm8k/"
output_eval_file = "Gsm8k_evaluation_results.txt"
num_fewshot = 5
batch_size = 1
max_new_tokens = 300

# In[ ]:
# This part remains the same: loading the base model for PPO fine-tuning
print(f"Loading tokenizer from local path: {local_model_path}")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
print(f"Loading model from local path: {local_model_path}")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=quant_config
)
print("Base Model initialization complete.")
base_model.eval()

# In[ ]:
print("load GSM8K dataset")
train_file = os.path.join(local_Gsm8k_path, "train.jsonl")
test_file = os.path.join(local_Gsm8k_path, "test.jsonl")
train_set = load_dataset('json', data_files=train_file, split='train')
test_set = load_dataset('json', data_files=test_file, split='train')
print(f"Successfully loaded {len(train_set)} training examples and {len(test_set)} test examples.")


# In[ ]:
def create_prompt(test_question):
    prompt = ""
    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"

    prompt += f"Question: {test_question}\n"
    prompt += "Answer:"
    return prompt


def extract_final_answer(model_output):
    match = re.search(r"####\s*([\d,]+\.?\d*)", model_output)
    if match:
        final_answer = match.group(1).replace(",", "")
        try:
            return float(final_answer)
        except ValueError:
            return None

    all_numbers = re.findall(r"[\d,]+\.?\d*", model_output)
    if all_numbers:
        last_number = all_numbers[-1].replace(",", "")
        try:
            return float(last_number)
        except ValueError:
            return None

    return None


def write_header(file_handle, model_path, num_fewshot):
    file_handle.write("--- GSM8K Evaluation Report ---\n")
    file_handle.write(f"Model Path: {model_path}\n")
    file_handle.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    file_handle.write(f"Number of Few-shot Examples: {num_fewshot}\n")
    file_handle.write("-" * 30 + "\n\n")


def write_question_result(file_handle, question_num, is_correct, question, model_output, predicted_answer,
                          ground_truth_answer):
    file_handle.write(f"--- Question #{question_num} ---\n")
    file_handle.write(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}\n")
    file_handle.write(f"Question: {question}\n")
    file_handle.write(f"\nModel Output:\n\n{model_output}\n\n")
    file_handle.write(f"\nPredicted Answer: {predicted_answer}\n")
    file_handle.write(f"Ground Truth Answer: {ground_truth_answer}\n")
    file_handle.write("=" * 30 + "\n\n")


def write_summary(file_handle, total, correct, accuracy):
    summary_string = (
        "\n--- FINAL SUMMARY ---\n"
        f"Total Questions Evaluated: {total}\n"
        f"Correct Predictions: {correct}\n"
        f"Accuracy: {accuracy:.2f}%\n"
    )
    file_handle.write(summary_string)
    return summary_string





# In[ ]:
few_shot_examples = train_set.select(range(num_fewshot))

# In[ ]:

print(f"Starting evaluation on {len(test_set)} test examples...")
correct_predictions = 0
total_predictions = 0
with open(general_path + output_eval_file, 'w', encoding='utf-8') as f:
    write_header(f, local_model_path, num_fewshot)

    for test_example in tqdm(test_set):
        total_predictions += 1
        question = test_example["question"]
        ground_truth_answer_text = test_example["answer"]
        ground_truth_answer = extract_final_answer(ground_truth_answer_text)

        if ground_truth_answer is None:
            print(f"no ground truth: {question}")
            f.write(f"Question #{total_predictions}: Skip, ground truth missing\n")
            continue

        prompt = create_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer_text = full_output[len(prompt):].strip()
        predicted_answer = extract_final_answer(model_answer_text)

        is_correct = False
        if predicted_answer is not None:
            if predicted_answer == ground_truth_answer:
                is_correct = True

        if is_correct:
            correct_predictions += 1

        write_question_result(
            f, total_predictions, is_correct, question,
            model_answer_text, predicted_answer, ground_truth_answer
        )

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
    else:
        accuracy = 0

    final_summary = write_summary(f, total_predictions, correct_predictions, accuracy)

print(final_summary)
print(f"Detailed report saved to '{output_eval_file}'")
