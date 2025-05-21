from collections import Counter
from tqdm import tqdm
from llm_class import PlainLLM, SparagiRDF
import torch
import argparse
import re
import pandas as pd
import time
torch.cuda.empty_cache()


FEW_SHOT_NUM = 0
REPEAT_FACTOR = 3
# regex to remove non-chars in necessary
regex = re.compile('[^a-zA-Z]')
llm = None

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

def getYesNoResponse(query):
    for i in range(REPEAT_FACTOR):
        responses = []
        try:
            # response = llm.generate(query + " Instruction: Respond with only 'yes' or 'no'. Do not include any other text or explanation.")  # for yes/no queries, append an extra instruction
            response = llm.generate(query, "yes/no", FEW_SHOT_NUM)
            response = response.replace("\n", " ").lower()
            response = regex.sub('', response)
            responses.append(response)
        except torch.cuda.OutOfMemoryError:
            print(bcolors.WARNING + f"CUDA out of memory error encountered, skipping query: '{query}'..." + bcolors.ENDC)
            response = "no response"
            responses.append(response)
    # Count occurrences of each option
    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    # conformity check
    if most_prominent_response not in ['yes', 'no']:
        print(bcolors.WARNING + f"Invalid response: {most_prominent_response}" + bcolors.ENDC)
        most_prominent_response = "no response"
    return most_prominent_response

def getRadioResponse(query):
    for i in range(REPEAT_FACTOR):
        responses = []
        try:
            # response = llm.generate(query + " \"Instruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text.\"")
            response = llm.generate(query, "radio", FEW_SHOT_NUM)
            response = response.replace("\n", " ").lower()
            response = response.replace(".", "")
            response = regex.sub('', response)
            # if response not in ["a","b","c","d","e"]:
            #     response = "no response"
            responses.append(response)
        except torch.cuda.OutOfMemoryError:
            print(bcolors.WARNING + f"CUDA out of memory error encountered, skipping query: '{query}'..." + bcolors.ENDC)
            response = "no response"
            responses.append(response)
    # Count occurrences of each option
    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    # conformity check
    if most_prominent_response not in ['a', 'b', 'c', 'd', 'e']:
        # mistral sometimes responds with not only the letter but also the option text
        # fix it manually
        if most_prominent_response[0] in ['a', 'b', 'c', 'd', 'e']:
            most_prominent_response = most_prominent_response[0]
        else:
            print(bcolors.WARNING + f"Invalid response: {most_prominent_response}" + bcolors.ENDC)
            most_prominent_response = "no response"
    return most_prominent_response


def getCheckboxResponse(query):
    for i in range(REPEAT_FACTOR):
        responses = []
        try:
            # response = llm.generate(query + " \"Instruction: Respond with only the letters (a-e) separated with comma, corresponding to the correct options. Do not include any explanation or additional text.\"")
            response = llm.generate(query, "checkbox", FEW_SHOT_NUM)
            response = response.replace("\n", " ").lower()
            response = response.replace(".", "")
            response = response.replace(" ", "")
            responses.append(response)
        except torch.cuda.OutOfMemoryError:
            print(bcolors.WARNING + f"CUDA out of memory error encountered, skipping query: '{query}'..." + bcolors.ENDC)
            response = "no response"
            responses.append(response)

    # Count occurrences of each option
    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    
    # conformity check
    tokens = most_prominent_response.split(',')
    for token in tokens:
        if token not in ['a', 'b', 'c', 'd', 'e','']:
            print(bcolors.WARNING + f"Invalid response: {most_prominent_response}" + bcolors.ENDC)
            most_prominent_response = "no response"
            break
    return most_prominent_response

def main():
    global llm, FEW_SHOT_NUM
    parser = argparse.ArgumentParser(description="Run an LLM with optional RAG functionality.")
    parser.add_argument("-query_dataset_path", type=str, help="Path of the query dataset to use")
    parser.add_argument("-query_result_path", type=str, help="Path of the output result")
    parser.add_argument("-qtype", type=str, help="Query type in file (if all queries have the same type only, for 3 column files)")
    parser.add_argument("-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to load")
    parser.add_argument("-quantize", type=int, default=4, help="Bits to quantize the model to. Options: [4,8]")
    parser.add_argument("-few_shot", type=int, default=0, help="Few-shot examples to add during inference.")
    args = parser.parse_args()

    llm_modelid = args.model
    FEW_SHOT_NUM = args.few_shot
    rdf_input_file = "/mnt/newdrive/data_files/SpaTex/CSZt.nt"
    llm = SparagiRDF(llm_modelid, rdf_input_file, args.quantize)
    print(bcolors.GREEN + "Using SPARAGI-RDF LLM" + bcolors.ENDC)

    # Load queries
    df = pd.read_csv(args.query_dataset_path)
    print(bcolors.GREEN + f"Running model {llm_modelid}" + bcolors.ENDC)

    output_path = args.query_result_path

    # Write header once at start
    col_num = len(df.columns)
    with open(output_path, "w") as f:
        if col_num == 3:
            f.write("type,query,truth,response\n")
        elif col_num == 2:
            f.write("query,truth,response\n")
        else:
            print(bcolors.RED + f"Invalid number of columns in file: {col_num}" + bcolors.ENDC)

    start = time.time()
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating queries..."):
        if col_num == 3:
            qtype = row['type']
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
        else:
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
            qtype = args.qtype

        print(bcolors.BLUE + f"Query Type: {qtype}, Query: {query}" + bcolors.ENDC)

        # Get response
        if qtype == "yes/no":
            response = getYesNoResponse(query)
        elif qtype == "radio":
            response = getRadioResponse(query)
        elif qtype == "checkbox":
            response = getCheckboxResponse(query)
        else:
            print(bcolors.WARNING + f"Unknown query type '{qtype}', ignoring..." + bcolors.ENDC)
            response = "no response"

        if isinstance(response, list):
            response = "; ".join(response)

        print(bcolors.RED + "response: " + str(response) + bcolors.ENDC)

        # Write single row as DataFrame
        if col_num == 3:
            pd.DataFrame([{
                "type": qtype,
                "query": query,
                "truth": truth,
                "response": response
            }]).to_csv(output_path, mode='a', index=False, header=False)
        else:
            pd.DataFrame([{
                "query": query,
                "truth": truth,
                "response": response
            }]).to_csv(output_path, mode='a', index=False, header=False)

    print(f"Generated and saved responses in {time.time() - start:.2f} seconds.")



if __name__=="__main__":
    main()
