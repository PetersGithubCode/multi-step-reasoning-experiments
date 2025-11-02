# In[ ]:
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import faiss
import os
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

# In[ ]:

main_path = "/home/peternicholson/Documents/"
base_model_path = "/home/peternicholson/Documents/gemma-2-9b-it/"
local_HpQA_path = "/home/peternicholson/Documents/base-HpQA/"
to_1_HpQA_path = "/home/peternicholson/Documents/1-HpQA/"
output_eval_file = "HpQA_evaluation_results.txt"

max_num = 4


# In[ ]:
def create_initial_prompt(f_question, f_max_num):
    updated_query_prompt_template = (f"<bos><start_of_turn>user\n"
                                     f"Please generate a search query enclosed by <search_query> QUERY </search_query> tags. I will allow you to make up to {f_max_num} sequential queries. I will provide search results in the following format:\n"
                                     f"QUERY → RESULT.\n"
                                     f"The question is: {f_question}\n<end_of_turn>")

    return updated_query_prompt_template


def create_answer_prompt(f_question, f_context):
    updated_answer_prompt_template = (
        f"<bos><start_of_turn>user You are a helpful assistant. Based ONLY on the provided context below, answer the user's question. If the answer cannot be found in the context, say \"The answer cannot be found in the provided context\" Context: {f_context} Question: {f_question} <end_of_turn><start_of_turn>model")

    return updated_answer_prompt_template


def prompt_a_model(f_prompt, f_tokenizer, f_model):
    inputs = f_tokenizer(f_prompt, return_tensors="pt").to(f_model.device)

    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask")
    }

    # dynamically set the length
    input_len = inputs["input_ids"].shape[1]
    eos_token_id = f_tokenizer.convert_tokens_to_ids("<end_of_turn>")

    outputs = f_model.generate(
        # **inputs,
        **model_inputs,
        max_new_tokens=20,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,

    )
    generated_tokens = outputs[0][input_len:]
    generated_query = f_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    generated_query = generated_query.split("<start_of_turn>model\n")[-1].strip()

    return generated_query


def search_query(search_query, index, context_list_joined, the_model):
    # SEARCH QUERY: converts the query into an embedding, searches for similarity amongst the embeddings
    # encode the query into an embedding
    query_embedding = the_model.encode([search_query], convert_to_numpy=True)
    # normalize embeddings for cosine similarity
    faiss.normalize_L2(query_embedding)
    # number of nearest neighbors to retrieve
    k = 1
    # perform search
    distances, indices = index.search(query_embedding, k)

    # retrieve and format results
    results = []
    for i in range(k):
        result = {
            'text': context_list_joined[indices[0][i]],
            'similarity_score': float(distances[0][i])
        }
        results.append(result)

    return result


def query_db(f_query, f_index, f_context_list, f_model):
    # search the query and return the response
    f_response = search_query(f_query, f_index, f_context_list, f_model)

    f_response_text = f_response['text']

    # format the response
    f_formatted_response = format_search_query_response(f_query, f_response_text)

    return f_formatted_response, f_response['text']


def format_search_query_response(f_query, f_context_response):
    f_response = (f"<start_of_turn>user\n"
                  f"{f_query} → {f_context_response}.\n"
                  f"<end_of_turn>\n")
    return f_response


def get_search_query_value(a_query):
    pattern = "<search_query>(.*?)</search_query>"
    match = re.search(pattern, a_query)
    if match:
        value = match.group(1)
        # print(value)
        return value
    else:
        print("No match found")
        return -1


def extract_supporting_facts(f_entry):
    titles_list = []
    supporting_facts = f_entry['supporting_facts']
    # extracts the titles
    combined_title_para = [title for title, _ in supporting_facts]
    titles_list.extend(combined_title_para)

    return titles_list


def generate_state_turn(f_type, f_max_num, f_entry):
    # first state
    if f_type == 'turn_seed':
        question = f_entry['question']
        question_prompt = create_initial_prompt(question, f_max_num)
        return question_prompt


def generate_state_from_query(f_query, f_index, f_context_list, f_model):
    context_response, unformatted_text = query_db(f_query, f_index, f_context_list, f_model)
    return context_response, unformatted_text


def generate_action(f_query_data, f_tokenizer, f_base_model):
    temp_string = str(f_query_data)
    just_for_query = temp_string + "<start_of_turn>model"
    question_response = prompt_a_model(just_for_query, f_tokenizer, f_base_model)
    # does it have a search query, return -1 if not query exists
    query_result = get_search_query_value(question_response)
    # format it
    question_response = "<start_of_turn>model\n" + str(question_response) + "<end_of_turn>\n"

    return question_response, query_result


def end_of_state(trajectory, question, golden_answer, supporting_facts_list, contexts_from_hops, f_local_HpQA_path):
    new_entry = {
        "question": question,
        "golden_answer": golden_answer,
        "supporting_facts_list": supporting_facts_list,
        "contexts_from_hops": contexts_from_hops,
        "trajectory": trajectory
    }
    log_entries(new_entry, f_local_HpQA_path)
    return new_entry


def process_questions(f_multi_step_entries, f_tokenizer, f_base_model):
    generated_data = []
    i = 0
    print("Num of entries to process: " + str(len(f_multi_step_entries)))
    for entry in f_multi_step_entries:
        print("i: " + str(i))
        i += 1
        query_counter = 0
        trajectory = ""
        contexts_from_hops = []
        turn = 'turn_seed'
        search_query_value = None
        question = entry['question']
        golden_answer = entry['answer']
        supporting_facts_list = extract_supporting_facts(entry)

        while True:
            # state 1
            if turn == 'turn_seed':
                question_prompt = generate_state_turn(turn, max_num, entry)
                turn = 'model'
                trajectory = trajectory + question_prompt

            if turn == 'user':
                # State 2 - Variant 1: has search query
                if search_query_value != -1:
                    if query_counter == max_num:
                        trajectory = trajectory + "<start_of_turn>user\nreached max queries<end_of_turn>\n<eos>"
                        new_entry = end_of_state(trajectory, question, golden_answer, supporting_facts_list,
                                                 contexts_from_hops, local_HpQA_path)
                        generated_data.append(new_entry)
                        break

                    else:
                        context_response, unformatted_text = generate_state_from_query(search_query_value, index,
                                                                                       context_list, model)
                        contexts_from_hops.append(unformatted_text)
                        query_counter += 1
                        turn = 'model'
                        trajectory = trajectory + context_response
                # State 2 - Variant 2: has no search query, answer on contexts gathered
                else:
                    # no need to set turn to 'end' it will update 'turn_seed' on next element
                    trajectory = trajectory + "<eos>"
                    new_entry = end_of_state(trajectory, question, golden_answer, supporting_facts_list,
                                             contexts_from_hops, local_HpQA_path)
                    generated_data.append(new_entry)
                    break

            # it's the model's turn
            if turn == 'model':
                question_response, search_query_value = generate_action(trajectory, f_tokenizer,
                                                                        f_base_model)
                trajectory = trajectory + question_response
                turn = 'user'
                if search_query_value == -1:
                    trajectory = trajectory + question_response
                    print("no search query in model response: " + str(question_response))
                    trajectory = trajectory + '<start_of_turn>user\nno search query in model response<end_of_turn>\n<eos>'
                    new_entry = end_of_state(trajectory, question, golden_answer, supporting_facts_list,
                                             contexts_from_hops, local_HpQA_path)
                    generated_data.append(new_entry)
                    break

    return generated_data


def log_entries(f_entry, f_path):
    with open(f_path + "base_line_log_of_entries.json", "a", encoding="utf-8") as file:
        json.dump(f_entry, file, indent=1)


# metrics functions
def reconstruct_titles_from_contexts(f_context, f_gold_titles_set):
    matched_titles = []
    # lower case each element
    normalized_context = [c.lower() for c in f_context]

    found_match = False
    for title in f_gold_titles_set:
        normalized_title = title.lower()

        for context_string in normalized_context:
            if normalized_title in context_string:
                matched_titles.append(title)
                break

    return matched_titles


def reconstruct_titles_from_contexts_for_precision(f_context, f_gold_titles_set):
    ranked_titles_for_ap = []

    gold_title_map = {title.lower(): title for title in f_gold_titles_set}

    for passage in f_context:
        normalized_passage = passage.lower()

        found_title_for_this_passage = None

        for lower_title, original_title in gold_title_map.items():
            if lower_title in normalized_passage:
                found_title_for_this_passage = original_title
                break

        ranked_titles_for_ap.append(found_title_for_this_passage)

    return ranked_titles_for_ap



def calculate_recall(retrieved_docs, gold_docs):
    found_list = reconstruct_titles_from_contexts(retrieved_docs, set(gold_docs))

    # to handle duplicates
    found_set = set(found_list)
    gold_set = set(gold_docs)

    # for testing in case no gold_docs
    if not gold_set:
        return 1.0

    true_positives = len(found_set.intersection(gold_set))
    # recall calc
    recall = true_positives / len(gold_set)

    return recall


def reconstruct_titles_from_contexts_ap(f_context, f_gold_titles_set):
    ranked_titles_for_ap = []
    gold_title_map = {title.lower(): title for title in f_gold_titles_set}
    for passage in f_context:
        normalized_passage = passage.lower()
        found_title_for_this_passage = None
        for lower_title, original_title in gold_title_map.items():
            if lower_title in normalized_passage:
                found_title_for_this_passage = original_title
                break
        ranked_titles_for_ap.append(found_title_for_this_passage)
    return ranked_titles_for_ap


def calculate_average_precision(retrieved_docs, gold_docs):
    reconstructed_ranked_list = reconstruct_titles_from_contexts_ap(retrieved_docs, set(gold_docs))

    gold_set = set(gold_docs)

    if not gold_set:
        return 1.0

    hits = 0
    precision_at_k = []

    for k, doc_title in enumerate(reconstructed_ranked_list, 1):
        if doc_title in gold_set:
            hits += 1
            precision = hits / k
            precision_at_k.append(precision)

    if not precision_at_k:
        return 0.0

    average_precision = sum(precision_at_k) / len(gold_set)

    return average_precision


def evaluate_retrieval_performance(list_of_samples, output_filepath, hop_num):
    all_recall_scores = []
    all_ap_scores = []

    print(f"Evaluating {len(list_of_samples)} samples...")

    for i, sample in enumerate(list_of_samples):
        gold_docs = sample.get("supporting_facts_list", [])
        retrieved_docs = sample.get("contexts_from_hops", [])

        # Handle cases where retrieval might have failed or returned nothing
        if not retrieved_docs:
            retrieved_docs = []

        #found_titles = evaluate_contexts(retrieved_docs, set(gold_docs))

        # Calculate metrics for the current sample
        recall = calculate_recall(retrieved_docs, gold_docs)
        ap = calculate_average_precision(retrieved_docs, gold_docs)

        # Store the scores
        all_recall_scores.append(recall)
        all_ap_scores.append(ap)

        print(f"  Sample {i + 1}: Recall={recall:.2f}, AP={ap:.2f}")

    # Calculate final average metrics across all samples
    average_recall = sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
    mean_average_precision = sum(all_ap_scores) / len(all_ap_scores) if all_ap_scores else 0.0

    print("\n--- Final Aggregated Results ---")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Mean Average Precision (MAP): {mean_average_precision:.4f}")

    # Prepare results for file output
    results = {
        "hop_num": str(hop_num),
        "total_samples_evaluated": len(list_of_samples),
        "average_recall": average_recall,
        "mean_average_precision": mean_average_precision
    }

    # Write the results to file
    try:
        with open(output_filepath + output_eval_file, 'a') as f:
            json.dump(results, f, indent=4)
        print(f"\nSuccessfully wrote results to '{output_filepath}'")
    except IOError as e:
        print(f"\nError writing to file: {e}")

    return results


def categorize_samples_by_hops(f_list_of_samples):

    # 1. Initialize four empty lists to store the categorized samples.
    samples_with_1_hop = []
    samples_with_2_hops = []
    samples_with_3_hops = []
    samples_with_4_hops = []

    # 2. Loop through the main list of samples.
    for sample in f_list_of_samples:
        # Use .get() with a default empty list to safely handle
        # cases where the 'contexts_from_hops' key might be missing.
        contexts = sample.get('contexts_from_hops', [])

        # 3. Determine the number of hops by the length of the list.
        num_hops = len(contexts)

        # 4. Append the entire sample to the correct list based on its hop count.
        if num_hops == 1:
            samples_with_1_hop.append(sample)
        elif num_hops == 2:
            samples_with_2_hops.append(sample)
        elif num_hops == 3:
            samples_with_3_hops.append(sample)
        elif num_hops == 4:
            samples_with_4_hops.append(sample)
        # Samples with 0 hops or more than 4 hops will be ignored.

    # 5. Return the four populated lists.
    return samples_with_1_hop, samples_with_2_hops, samples_with_3_hops, samples_with_4_hops



# In[ ]:
# Load the HotPotQA seed questions data
output_file = local_HpQA_path + "1-HpQA_seed_questions.json"
with open(output_file, 'r', encoding='utf-8') as f:
    seed_questions_hotpot = json.load(f)
print(f"Selected {len(seed_questions_hotpot)} multi-step entries loaded from {output_file}.")

# In[ ]:
# Load the context_list seed questions data
output_file = local_HpQA_path + "1-HpQA_context_list.json"
with open(output_file, 'r', encoding='utf-8') as f:
    context_list = json.load(f)
print("loaded the context_list")

# In[ ]:
# embedding model used
model_name = 'all-mpnet-base-v2'
# use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_b = local_HpQA_path + '/embeddings'
embeddings_file = os.path.join(dir_b, 'embeddings_float16.npy')
index_file = os.path.join(dir_b, 'faiss_index.bin')
k = 1

if not os.path.exists(index_file):
    raise FileNotFoundError(f"FAISS index not found at {index_file}")
index = faiss.read_index(index_file)
print(f"Loaded FAISS index with {index.ntotal} vectors")
model = SentenceTransformer(model_name).to(device)

# In[ ]:
# load the base model in quantized to 4bit use as inference model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto",
)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

# In[ ]:
print("number of questions to process: " + str(len(seed_questions_hotpot)))
subset = seed_questions_hotpot[:3000]
list_of_generated_samples = process_questions(subset, base_tokenizer, base_model)

# In[ ]:
one, two, three, four = categorize_samples_by_hops(list_of_generated_samples)
evaluate_retrieval_performance(one, local_HpQA_path, 1)
evaluate_retrieval_performance(two, local_HpQA_path, 2)
evaluate_retrieval_performance(three, local_HpQA_path, 3)
evaluate_retrieval_performance(four, local_HpQA_path, 4)
# In[ ]:
print("finished")

