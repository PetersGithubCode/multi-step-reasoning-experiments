#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json
import re
import ollama
from sympy import sympify, SympifyError, symbols, Eq, solve, Symbol, Mul, Float
from sympy.physics.units import minute, hour
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# In[ ]:
# GSM8K training data
seed_questions_gsm8k = []
seed_answers_gsm8k = []

# specify directories to work with
dir_a = '/home/peternicholson/Documents/1-A-Gsm8k/'
train_data_file = "grade-school-math-master/grade_school_math/data/train.jsonl"
base_model_path = "/home/peternicholson/Documents/gemma-2-9b-it"

# In[ ]:
path_to_dataset = dir_a + train_data_file


def load_dataset_Gsm8k(f_path_to_file):
    f_seed_questions_gsm8k = []
    try:
        with open(f_path_to_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace from the end of the line
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        data_object = json.loads(stripped_line)
                        f_seed_questions_gsm8k.append(data_object)
                    except json.JSONDecodeError as e:
                        print(f"error decoding JSON from line: '{stripped_line}'")
                        print(f"error message: {e}")

    except FileNotFoundError:
        print(f"error: The file was not found at {dir_a + train_data_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return f_seed_questions_gsm8k


seed_questions_gsm8k = load_dataset_Gsm8k(path_to_dataset)
print("length of dataset: " + str(len(seed_questions_gsm8k)))


# In[ ]:
def extract_questions_and_answers_gsm8k(f_seed_questions_gsm8k):
    # get all contexts as paragraphs in a list of strings
    f_answer_list = []
    f_question_list = []
    # loop the whole set
    for item in f_seed_questions_gsm8k:
        # extract the question
        f_question_list.append(item['question'])
        # extract the answer
        f_answer_list.append(item['answer'])

    return f_question_list, f_answer_list


def build_seed_query_math_exp(f_initial_question, f_max_num):
    f_query = (f"<start_of_turn>user\n"
               f"Please help me answer the following question in just a few words. If you think it would help to to use a calculator, please generate a mathematical query enclosed by <math_exp> MATH EXP </math_exp> tags. Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to {f_max_num} sequential queries before answering the question. Please do not repeat queries you have already issued, as this is a waste of time. I will provide results in the following format:\n"
               f"QUERY → RESULT\n"
               f"Once you have enough information, generate an answer enclosed by <answer>ANSWER</answer> tags. Please either issue a search query or answer the question, but not both. The question is: {f_initial_question}\n"
               f"<end_of_turn>\n")
    return f_query


def post_llm(f_query, f_model):
    f_messages = [
        {"role": "user", "content": f_query}
    ]
    f_response = ollama.chat(model=f_model, messages=f_messages)
    f_assistant_response = f_response['message']['content']
    # print(f"Assistant: {assistant_response}")

    return f_assistant_response


def format_llm_query_response(f_query):
    # clean the response:
    f_query = f_query.replace("\n", "")
    f_formatted_response = (f"<start_of_turn>model\n"
                            f"{f_query}"
                            f"<end_of_turn>\n")
    return f_formatted_response


def has_math_exp_tags(f_to_check):
    # response from the query can be a dictionary, extract string from the dict
    if isinstance(f_to_check, dict):
        f_to_check = f_to_check.get('text', '')

    f_pattern = "<math_exp>(.*?)</math_exp>"
    # re.search only works on strings
    f_match = re.search(f_pattern, f_to_check)
    if f_match:
        return True
    else:
        return False


def has_answer_tags(f_to_check):
    # response from the query can be a dictionary, extract string from the dict
    if isinstance(f_to_check, dict):
        f_to_check = f_to_check.get('text', '')

    f_pattern = "<answer>(.*?)</answer>"
    # re.search only works on strings
    f_match = re.search(f_pattern, f_to_check)
    if f_match:
        return True
    else:
        return False


def format_numerical_result_response(f_query, f_numerical_result):
    f_formatted_response = (f"<start_of_turn>user\n"
                            f"{f_query} → {f_numerical_result}.\n"
                            f"<end_of_turn>\n")
    return f_formatted_response


def format_numerical_result(f_numerical_result):
    f_formatted_response = (f"<start_of_turn>user\n"
                            f"{f_numerical_result}"
                            f"<end_of_turn>\n")
    return f_formatted_response


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
        #**inputs,
        **model_inputs,
        max_new_tokens=50,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,

    )
    generated_tokens = outputs[0][input_len:]
    f_gen_answer = f_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return f_gen_answer


def query_the_llm(f_prompt, f_model, f_tokenizer):
    f_response = prompt_a_model(f_prompt, f_tokenizer, f_model)
    # post the query, get the response, update the trajectories
    # f_response = post_llm(f_query, f_model)
    # format the response
    f_formatted_response = format_llm_query_response(f_response)

    return f_response, f_formatted_response


def get_math_exp_value(f_query):
    # get the 1st math tag if there is more than one
    f_pattern = "<math_exp>(.*?)</math_exp>"
    f_match = re.search(f_pattern, f_query)
    if f_match:
        f_value = f_match.group(1)
        return f_value
    else:
        print("no math tag found")
        return None


def create_unit_map(equivalences):
    normalization_map = {}
    for canonical, variants in equivalences.items():
        normalization_map[canonical] = canonical
        for variant in variants:
            normalization_map[variant] = canonical
    return normalization_map


def clean_expression(f_string, f_unit_map, f_multi_word_map):
    f_cleaned = f_string.strip().rstrip("=?").strip()

    #Handle percentages
    f_cleaned = re.sub(r'(\d+\.?\d*|\.\d+)\s*%', lambda m: str(float(m.group(1)) / 100.0), f_cleaned)

    #cleaning rules
    f_cleaned = re.sub(r'[\$€£¥]\s*', '', f_cleaned)
    f_cleaned = re.sub(r'(?<=\d),(?=\d)', '', f_cleaned)
    f_cleaned = re.sub(r'\b(\d+)\s+(\d+/\d+)\b', r'(\1 + \2)', f_cleaned)
    f_cleaned = f_cleaned.replace(' and ', ' + ')
    f_cleaned = re.sub(r'\s+per\s+', ' / ', f_cleaned)
    for phrase, canonical in sorted(f_multi_word_map.items(), key=lambda item: len(item[0]), reverse=True):
        f_cleaned = f_cleaned.replace(phrase, canonical)
    f_words_to_replace = sorted(f_unit_map.keys(), key=len, reverse=True)
    if f_words_to_replace:
        pattern = r'\b(' + '|'.join(re.escape(word) for word in f_words_to_replace) + r')\b'
        f_cleaned = re.sub(pattern, lambda m: f_unit_map[m.group(0)], f_cleaned)
    f_cleaned = re.sub(r'([*+/()^=])', r' \1 ', f_cleaned)  # Added '=' and '^' as operators to space around
    f_cleaned = re.sub(r'(?<=[a-zA-Z0-9)])\s+(?=[a-zA-Z(])', ' * ', f_cleaned)
    f_cleaned = re.sub(r'(?<=[a-zA-Z0-9)])(?=[(])', ' * ', f_cleaned)
    f_cleaned = re.sub(r'(?<=[0-9])(?=[a-zA-Z])', ' * ', f_cleaned)
    f_cleaned = re.sub(r'\s+', ' ', f_cleaned).strip()
    return f_cleaned


def parser_for_sympy_equations(f_string, f_unit_map, f_multi_word_map, f_locals_dict, f_conversions):
    f_original_string = f_string.strip()

    if f_original_string.lower().startswith('solve(') and f_original_string.endswith(')'):
        f_content = f_original_string[len('solve('):-1]
        f_equation_strings = [s.strip() for s in f_content.split('and')]
        f_equations = []
        for eq_str in f_equation_strings:
            if '=' in eq_str:
                lhs_str, rhs_str = eq_str.split('=', 1)
                #variables in the equations
                all_vars = {**f_locals_dict, 'y': Symbol('y')}  # Add y temporarily
                cleaned_lhs = clean_expression(lhs_str, f_unit_map, f_multi_word_map)
                cleaned_rhs = clean_expression(rhs_str, f_unit_map, f_multi_word_map)
                lhs_expr = sympify(cleaned_lhs, locals=all_vars, evaluate=False)
                rhs_expr = sympify(cleaned_rhs, locals=all_vars, evaluate=False)
                f_equations.append(Eq(lhs_expr, rhs_expr))
        if f_equations:
            return solve(f_equations)
        else:
            raise ValueError("Could not parse equations inside solve().")

    if '=' in f_original_string and '?' in f_original_string:
        lhs_str, _ = f_original_string.split('=', 1)
        cleaned_lhs = clean_expression(lhs_str, f_unit_map, f_multi_word_map)
        lhs_expr = sympify(cleaned_lhs, locals=f_locals_dict, evaluate=True)
        converted_expr = lhs_expr.subs(f_conversions)
        return converted_expr.doit()

    f_cleaned_string = f_original_string.rstrip("=").strip()
    if ':' in f_cleaned_string:
        _, math_part = f_cleaned_string.split(':', 1)
        f_cleaned_string = math_part.strip()

    if '=' in f_cleaned_string:
        lhs_str, rhs_str = f_cleaned_string.split('=', 1)
        cleaned_lhs = clean_expression(lhs_str, f_unit_map, f_multi_word_map)
        cleaned_rhs = clean_expression(rhs_str, f_unit_map, f_multi_word_map)
        lhs_expr = sympify(cleaned_lhs, locals=f_locals_dict, evaluate=False)
        rhs_expr = sympify(cleaned_rhs, locals=f_locals_dict, evaluate=False)
        return Eq(lhs_expr, rhs_expr)
    else:
        f_cleaned_expr = clean_expression(f_cleaned_string, f_unit_map, f_multi_word_map)
        return sympify(f_cleaned_expr, locals=f_locals_dict, evaluate=False)


def process_tool_call(f_expression, f_unit_normalization_map, f_multi_word_variables, f_locals_for_sympy,
                      f_unit_conversions, f_trajectories):
    # check and format the expression so that it doesn't fail in sympy
    try:
        # print(f"Original: '{expression.strip()}'")
        f_sympy_obj = parser_for_sympy_equations(
            f_expression,
            f_unit_normalization_map,
            f_multi_word_variables,
            f_locals_for_sympy,
            f_unit_conversions
        )
        if isinstance(f_sympy_obj, Eq):
            print(f"SymPy solve: {f_sympy_obj}")
            f_tool_call_result = solve(f_sympy_obj)
            # print(f"Solution object: {outcome}\n")
            f_formatted_response = format_numerical_result_response(str(f_expression), str(f_tool_call_result))
            # append returned context to the history
            f_trajectories = f_trajectories + f_formatted_response
            return f_trajectories, True

        else:
            # This branch handles simple expressions, conversion results, and solve() results
            f_tool_call_result = sympify(str(f_sympy_obj))
            print(f"SymPy sympify: {str(f_sympy_obj)}\n")
            f_formatted_response = format_numerical_result_response(str(f_expression), str(f_tool_call_result))
            # append returned context to the history
            f_trajectories = f_trajectories + f_formatted_response
            return f_trajectories, True

    except Exception as f_e:
        # write error to the bad lines file, but also pass back the response to be written as a trajectory
        f_error_string = f"error in tool call'{f_expression.strip()}': {f_e}\n"
        print(f_error_string)
        f_formatted_response = format_numerical_result_response(str(f_expression), str(f_error_string))
        # add end of state to the f_formatted_response
        f_formatted_response = f_formatted_response + "<eos>\n"
        # append returned context to the history
        f_trajectories = f_trajectories + f_formatted_response
        return f_trajectories, False


def remove_irrelevant_words(f_string: str):
    s = f_string.strip()
    s = re.sub(r'^(what is|calculate|find|compute|determine)\b[\s:]*', '', s, flags=re.IGNORECASE)
    s = s.rstrip("?")
    return s.strip()


def write_out_file_json_dump(f_name, f_action, f_list):
    with open(f_name, f_action, encoding="utf-8") as f_file:
        json.dump(f_list, f_file, indent=1)


def write_out_file(f_name, f_action, f_list):
    with open(f_name, f_action) as file:
        file.write(f_list + "\n")


# In[ ]:

q_gsm8k, a_gsm8k = extract_questions_and_answers_gsm8k(seed_questions_gsm8k)
print("length of questions: " + str(len(q_gsm8k)))
# print(q_gsm8k[0])
# print(a_gsm8k[0])
# write the questions to file
write_out_file_json_dump(dir_a + '1-q_gsm8k.txt', 'w', q_gsm8k)
# write the answers to file
write_out_file_json_dump(dir_a + '1-a_gsm8k.txt', 'w', a_gsm8k)

# In[ ]:
unit_equivalences = {
    'snake': ['snakes'],
    'bird': ['birds'],
    'jaguar': ['jaguars'],
    'month': ['months'],
    'kg': ['kgs', 'kilogram', 'kilograms'],
    'minute': ['minutes'],
    'hour': ['hours'],
    'quarter': ['quarters'], 'nickel': ['nickels'], 'inch': ['inches'], 'fish': [],
    'hour': ['hours'], 'minute': ['minutes'], 'movie': ['movies'], 'issue': ['issues'],
    'mile': ['miles'], 'fruit': ['fruits'], 'egg': ['eggs'], 'week': ['weeks'],
    'word': ['words'], 'ml': [], 'gram': ['grams'], 'bale': ['bales'], 'acre': ['acres'],
    'year': ['years']
}
unit_normalization_map = create_unit_map(unit_equivalences)
multi_word_variables = {
    "Benedict's house": "Benedicts_house",
    "types of fruit": "fruit_types",
    "total eggs": "total_eggs",
    "adult eggs": "adult_eggs",
    "total bales harvested this year": "total_bales_harvested_this_year"
}
unit_conversions = {
    Symbol('hour'): 60 * Symbol('minute'),
}

# Make sure all potential variables are known
all_known_symbols = ['x', 'y']
locals_for_sympy = {name: Symbol(name) for name in set(all_known_symbols)}

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
def generate_trajectories(f_q_gsm8k, q_lines, f_model, f_tokenizer):
    trajectories_generated = []
    # up to max number of tool calls
    max_num = 10

    y = 0
    for f_question in f_q_gsm8k:
        with open(q_lines, 'a') as file:
            file.write(f_question + "\n")

        multi_step_questions_gsm8k = []
        tool_call = False
        trajectories = ""
        answered = False
        formatted_response = ""
        response = ""
        seed = True
        i = 0
        error = False
        x = 0
        turn = "model"
        already_insisted = False
        while True:
            # beginning - ask the seed question - the original prompt (pg 3)
            if seed:
                # get the question
                initial_question = f_question
                # build the seed query
                seed_query = build_seed_query_math_exp(initial_question, max_num)
                # update history
                trajectories = seed_query
                response, formatted_response = query_the_llm(seed_query, f_model, f_tokenizer)
                # once the model has responded, it's now the users turn
                turn = "user"
                # check response for tool call
                tool_call = has_math_exp_tags(response)
                # check response for answer tag
                answered = has_answer_tags(response)
                # sometimes response had both, answered wins
                if answered:
                    query_call = False
                print("seed response from llm: " + response)
                # update history
                trajectories = trajectories + formatted_response
                seed = False

            if turn == "user":
                if tool_call:
                    # if it's a tool call
                    # set tool call to false
                    tool_call = False
                    # increment the tool call
                    i += 1
                    print("i: " + str(i))
                    # remove the 1st instance of math tags value for tool call
                    expression = get_math_exp_value(response)
                    # remove irrelevant words from expression if any
                    expression = remove_irrelevant_words(expression)
                    # process the tool_call
                    trajectories, resp_sympy = process_tool_call(expression, unit_normalization_map,
                                                                 multi_word_variables,
                                                                 locals_for_sympy, unit_conversions, trajectories)
                    # once you get a response it's now the models turns
                    turn = "model"
                    if not resp_sympy:
                        # instead of logging it save the response to the trajectory and save it
                        print("ERROR processing sympy")
                        new_entry = {
                            "question": f_question,
                            "trajectories": trajectories
                        }
                        multi_step_questions_gsm8k.append(new_entry)
                        error = True
                        break
                # if it's an answer
                elif answered:
                    print("answer tag encountered")
                    # make answered false
                    answered = False
                    # append close to trajectory
                    trajectories = trajectories + "<eos>\n"
                    new_entry = {
                        "question": f_question,
                        "trajectories": trajectories
                    }
                    multi_step_questions_gsm8k.append(new_entry)
                    break

                # if there is no tool call and no answer then ask the model to respond properly
                # log it
                else:
                    # append close to trajectory
                    trajectories = trajectories + "<eos>\n"
                    new_entry = {
                        "question": f_question,
                        "trajectories": trajectories
                    }
                    multi_step_questions_gsm8k.append(new_entry)
                    break


            if turn == "model":
                # then its a query
                # do query to llm
                response, formatted_response = query_the_llm(seed_query, f_model, f_tokenizer)
                # once the model responds its now the users turn
                turn = "user"
                print("seed response from llm: " + response)
                # check response for tool call
                tool_call = has_math_exp_tags(response)
                # check response for answer tag
                answered = has_answer_tags(response)
                # sometimes response had both, answered wins
                if answered:
                    query_call = False
                # update history
                trajectories = trajectories + formatted_response
                #to prevent tool calls going beyond the max_num
                if i == max_num:
                    # append close to trajectory
                    trajectories = trajectories + "<eos>\n"
                    new_entry = {
                        "question": f_question,
                        "trajectories": trajectories
                    }
                    multi_step_questions_gsm8k.append(new_entry)
                    break

            if i == max_num + 1 and answered is False:
                # append close to trajectory
                trajectories = trajectories + "<eos>\n"
                new_entry = {
                    "question": f_question,
                    "trajectories": trajectories
                }
                multi_step_questions_gsm8k.append(new_entry)
                break

            if x == 50 or error is True:
                # append close to trajectory
                trajectories = trajectories + "<eos>\n"
                new_entry = {
                    "question": f_question,
                    "trajectories": trajectories
                }
                multi_step_questions_gsm8k.append(new_entry)
                break

            x += 1

        # update after ending the while
        write_out_file_json_dump(dir_a + '1-Gsm8k_trajectories_no_filtering.json', 'a', multi_step_questions_gsm8k)
        print("processed record: " + str(y))
        y += 1


    return


# In[ ]:
q_lines = dir_a + '1-Gsm8k_question_lines.txt'
generate_trajectories(q_gsm8k, q_lines, base_model, base_tokenizer)


# In[ ]:
print("finished")
