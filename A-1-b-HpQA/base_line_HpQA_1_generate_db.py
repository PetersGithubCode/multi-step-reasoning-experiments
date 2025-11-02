# In[ ]:
import json
import random
import re
import ollama
from tqdm import tqdm
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import time
import os
import psutil

# In[ ]:
#-----------------------------------------------------------
#1. LOAD data set and prepare
#-----------------------------------------------------------

# In[2]:
dir_a = '/home/peternicholson/Documents/base-HpQA/'
main_dir = '/home/peternicholson/Documents/'

# In[ ]:
# Load the HotPotQA training data
# with open('/Users/rdd2/Documents/llm-envionments/Project1/hotpot_train_v1.1.json', 'r') as f:
with open(
        main_dir + 'hotpot_train_v1.1.json',
        'r') as f:
    hotpot_data = json.load(f)

# filter out the easy  (top of pg 4 swirl paper)
multi_step_entries = [entry for entry in hotpot_data if
                      entry.get('level') != 'easy' and len(entry.get('supporting_facts', [])) > 1]
print(f"Found {len(multi_step_entries)} multi-step entries.")
# select 10,000 randoms
num_samples = min(10000, len(multi_step_entries))
seed_questions_hotpot = random.sample(multi_step_entries, num_samples)

# save it
output_file = dir_a + "1-HpQA_seed_questions.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(seed_questions_hotpot, f, indent=2)

print(f"Selected {len(seed_questions_hotpot)} multi-step entries and saved to {output_file}.")


# In[ ]:
# FUNCTIONS A: preparation data related functions
def extract_contexts(seed_questions_hotpot):
    # get all contexts as paragraphs in a list of strings
    context_list = []
    question_list = []
    # loop the whole set
    for item in seed_questions_hotpot:
        # extract the context paragphs and join into a string "context_paragraph"
        combined_contexts = [
            {
                'title': context[0],  # Extract the title
                'context': ' '.join(context[1])  # Join sentences into a single string
            }
            for context in item['context']
        ]

        # print(combined_context_para)
        # store the context_paragraph into the list of contexts
        context_list.append(combined_contexts)

        # extract the question also
        question_list.append(item['question'])

    return question_list, context_list


def list_of_contexts(context_group):
    list_of_concat_contexts = []
    for i in range(len(context_group)):
        for j in range(len(context_group[i])):
            context = ""
            context = context_group[i][j]
            # contexts_concatonated = contexts_concatonated + value
            list_of_concat_contexts.append(context)
        # list_of_concat_contexts.append(contexts_concatonated)

    return list_of_concat_contexts


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


def get_search_query_value(a_query):
    pattern = "<search_query>(.*?)</search_query>"
    match = re.search(pattern, a_query)
    if match:
        value = match.group(1)
        # print(value)
        return value
    else:
        print("No match found")
        return None


# In[ ]:
# prepare HotpotQA data
# extract the contexts
questions_list, contexts_seperated = extract_contexts(seed_questions_hotpot)
# convert paragraphs into lists [i][j] (i is each question)(j is ech context concatonated together)
# context_list_joined = join_contexts(context_list)
context_list = list_of_contexts(contexts_seperated)

# In[ ]:
# FOR TESTING
# print(len(context_list))
# print(context_list[1])
max_length = max(len(item) for item in context_list)
# print(max_length)
# get the longest string
longest_string = max(context_list, key=len)
print(longest_string)

# In[ ]:
# save it
output_file = dir_a + "1-HpQA_context_list.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(context_list, f, indent=2)

# In[ ]:
#-----------------------------------------------------------
#2. GENERATE DB of vector embeddings and index
#-----------------------------------------------------------
# In[ ]:
batch_size = 64
chunk_size = 5000
model_name = 'all-mpnet-base-v2'
#use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = dir_a + '/embeddings'
os.makedirs(output_dir, exist_ok=True)
embeddings_file = os.path.join(output_dir, 'embeddings_float16.npy')
index_file = os.path.join(output_dir, 'faiss_index.bin')

# In[ ]:
# load existing embeddings
if os.path.exists(embeddings_file) and os.path.exists(index_file):
    print(f"Existing normalized embeddings in file: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if np.any(~np.isfinite(embeddings)):
        print("Loaded embeddings: NaN or inf values. replaced with zeros.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    index = faiss.read_index(index_file)
    print(f"Loaded FAISS index: {index.ntotal} vectors")
# or create new embeddings
else:
    # Initialize model
    model = SentenceTransformer(model_name).to(device)

    # Process embeddings in chunks
    embeddings_list = []
    index = None
    #to show how long for processing
    total_start_time = time.time()

    for i in range(0, len(context_list), chunk_size):
        chunk_start_time = time.time()
        chunk = context_list[i:i + chunk_size]

        try:
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    chunk_embeddings = model.encode(
                        chunk,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        device=device,
                        num_workers=0,
                        max_length=512
                    )
            else:
                chunk_embeddings = model.encode(
                    chunk,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    device=device,
                    num_workers=0,  # got semaphore leaks when > 0 on mac
                    max_length=512  # Truncate long texts
                )

            # Convert to float32 for normalization
            chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)

            # Check for NaN/inf
            if np.any(~np.isfinite(chunk_embeddings)):
                print(f"Warning: Chunk {i // chunk_size} NaN or inf values found. Replacing with zeros.")
                chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize embeddings
            faiss.normalize_L2(chunk_embeddings)

            # Save normalized embeddings as float16
            chunk_file = os.path.join(output_dir, f"normalized embeddings chunk {i // chunk_size} float16.npy")
            np.save(chunk_file, chunk_embeddings.astype(np.float16))
            print(f"Saved normalized embeddings for chunk {i // chunk_size} to {chunk_file}")

            # Initialize or update FAISS index
            if index is None:
                dimension = chunk_embeddings.shape[1]
                cpu_index = faiss.IndexFlatIP(dimension)
                faiss.omp_set_num_threads(4)
                if device.type == "cuda":
                    try:
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                        print("Using GPU index")
                    except Exception as e:
                        print("Failed to init FAISS on GPU, using CPU:", e)
                        index = cpu_index
                else:
                    index = cpu_index

            index.add(chunk_embeddings)

            # Save FAISS index
            faiss.write_index(faiss.index_gpu_to_cpu(index) if device.type == "cuda" else index, index_file)
            print(f"Updated FAISS index saved to {index_file}")

            embeddings_list.append(chunk_embeddings)
            #Release memory
            del chunk_embeddings
            print(f"Chunk {i // chunk_size} processed in {time.time() - chunk_start_time:.2f} seconds")

            #Check memory usage
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"Memory usage: {mem_info.rss / 1e9:.2f} GB")

        except Exception as e:
            print(f"Error processing chunk {i // chunk_size}: {e}")
            raise

    #Concatenate and save all embeddings as float16
    embeddings = np.concatenate(embeddings_list)
    np.save(embeddings_file, embeddings.astype(np.float16))
    print(f"Saved all normalized embeddings to {embeddings_file}")

    print(f"Total processing time: {time.time() - total_start_time:.2f} seconds")
    # Free memory
    del embeddings_list

# Use index and embeddings for further processing
print(f"FAISS index contains {index.ntotal} vectors")

# In[ ]:








