from collections import Counter
from tqdm import tqdm
import torch
import argparse
import re
import pandas as pd
import time

from index.rdf_graph_index import RDFGraphIndex
from CG.rdf_composition import RDF_Composition
from NER.spacy_ner import SpacyNER
from CG.composition_matrix import ApproximateCM

from fewshot.ann_similar import ANNFewShot

from utils import bcolors

from model_module import PlainLLM, SpaRAGraph

torch.cuda.empty_cache()

REPEAT_FACTOR = 1
# regex to remove non-chars in necessary
regex = re.compile('[^a-zA-Z]')
llm = None

def getYesNoResponse(query):
    for i in range(REPEAT_FACTOR):
        responses = []
        try:
            response = llm.runQuery(query, "BINARY")
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
            response = llm.runQuery(query, "MULTICLASS")
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
            response = llm.runQuery(query, "MULTILABEL")
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
    global llm
    parser = argparse.ArgumentParser(description="Run an LLM with optional RAG functionality.")
    parser.add_argument("-rdf_dataset_path", type=str, default="/mnt/newdrive/data_files/SpaTex/CSZt.nt", help="Path of the RDF dataset to use")
    parser.add_argument("-query_dataset_path", type=str, help="Path of the query dataset to use")
    parser.add_argument("-query_result_path", type=str, help="Path of the output result")
    parser.add_argument("-qtype", type=str, help="Query type [yes/no,radio,checkbox] in query file (if all queries have the same type only, for 3 column files)")
    parser.add_argument("-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to load")
    parser.add_argument("-mode", type=str, default="SPARAGRAPH", help="[PLAIN,SPARAGRAPH]")
    parser.add_argument("-quantize", type=int, default=4, help="Bits to quantize the model to. Options: [4,8]")
    parser.add_argument("-few_shot", type=int, default=0, help="Few-shot examples to add during inference.")
    args = parser.parse_args()

    print(args)

    llm_modelid = args.model
    rdf_input_file = args.rdf_dataset_path
    llm = None
    if args.mode == "PLAIN":
        # plain LLM
        llm = PlainLLM(llm_modelid, args.quantize)
    elif args.mode == "SPARAGRAPH":
        # with sparagraph
        llm = SpaRAGraph(llm_modelid, 
                         RDFGraphIndex(rdf_input_file, RDF_Composition(ApproximateCM())) , 
                         SpacyNER(), 
                         ANNFewShot(),
                         args.quantize, 
                         args.few_shot)
        
    # Load queries
    df = pd.read_csv(args.query_dataset_path)
    # setup output
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

        if index % 5 == 0:
            torch.cuda.empty_cache()
            
        if index >= 753:
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
