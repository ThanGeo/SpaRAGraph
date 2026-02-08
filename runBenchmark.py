
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps debug
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from collections import Counter
from tqdm import tqdm
import torch
import argparse
import re
import pandas as pd
import time
import gc

from index.rdf_graph_index import RDFGraphIndex
from index.faiss_index import FAISSIndex
from CG.rdf_composition import RDF_Composition
from NER.spacy_ner import SpacyNER
from CG.composition_matrix import ApproximateCM

from fewshot.ann_similar import ANNFewShot

from utils import bcolors

from model_module import PlainLLM, SpaRAGraph, BaseRAG

# Clear cache at startup
torch.cuda.empty_cache()
gc.collect()

REPEAT_FACTOR = 1
regex = re.compile('[^a-zA-Z]')
llm = None

def safe_cuda_operation(func, *args, max_retries=2, **kwargs):
    """Wrapper for CUDA operations with automatic retry on error"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                print(bcolors.RED + f"CUDA error (attempt {attempt+1}/{max_retries}): {str(e)}" + bcolors.ENDC)
                
                # Aggressive cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                if attempt < max_retries - 1:
                    print(bcolors.WARNING + "Retrying after cleanup..." + bcolors.ENDC)
                    time.sleep(1)  # Brief pause
                else:
                    print(bcolors.RED + "Max retries reached, skipping query" + bcolors.ENDC)
                    raise
            else:
                raise

def getYesNoResponse(query):
    responses = []
    
    for i in range(REPEAT_FACTOR):
        try:
            response = safe_cuda_operation(llm.runQuery, query, "BINARY")
            response = response.replace("\n", " ").lower()
            response = regex.sub('', response)
            responses.append(response)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(bcolors.WARNING + f"Error encountered, skipping query: '{query}'... ({str(e)[:100]})" + bcolors.ENDC)
            response = "no response"
            responses.append(response)
            break
    
    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    
    if most_prominent_response not in ['yes', 'no']:
        print(bcolors.WARNING + f"Invalid response: {most_prominent_response}" + bcolors.ENDC)
        most_prominent_response = "no response"
    
    return most_prominent_response

def getRadioResponse(query):
    responses = []
    
    for i in range(REPEAT_FACTOR):
        try:
            response = safe_cuda_operation(llm.runQuery, query, "MULTICLASS")
            response = response.replace("\n", " ").lower()
            response = response.replace(".", "")
            response = regex.sub('', response)
            responses.append(response)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(bcolors.WARNING + f"Error encountered, skipping query: '{query}'... ({str(e)[:100]})" + bcolors.ENDC)
            response = "no response"
            responses.append(response)
            break
    
    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    
    if most_prominent_response not in ['a', 'b', 'c', 'd', 'e']:
        if most_prominent_response and most_prominent_response[0] in ['a', 'b', 'c', 'd', 'e']:
            most_prominent_response = most_prominent_response[0]
        else:
            print(bcolors.WARNING + f"Invalid response: {most_prominent_response}" + bcolors.ENDC)
            most_prominent_response = "no response"
    
    return most_prominent_response

def getCheckboxResponse(query):
    responses = []
    
    for i in range(REPEAT_FACTOR):
        try:
            response = safe_cuda_operation(llm.runQuery, query, "MULTILABEL")
            response = response.replace("\n", " ").lower()
            response = response.replace(".", "")
            response = response.replace(" ", "")
            responses.append(response)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(bcolors.WARNING + f"Error encountered, skipping query: '{query}'... ({str(e)[:100]})" + bcolors.ENDC)
            response = "no response"
            responses.append(response)
            break

    counts = Counter(responses)
    most_prominent_response = counts.most_common(1)[0][0]
    
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
    parser.add_argument("-mode", type=str, default="SPARAGRAPH", help="[PLAIN,SPARAGRAPH,FAISS]")
    parser.add_argument("-quantize", type=int, default=4, help="Bits to quantize the model to. Options: [4,8]")
    parser.add_argument("-few_shot", type=int, default=0, help="Few-shot examples to add during inference.")
    parser.add_argument("-k", type=int, default=1, help="Number of tuples to retrieve for context generation. Works for the following: [FAISS]")
    args = parser.parse_args()

    print(args)

    llm_modelid = args.model
    rdf_input_file = args.rdf_dataset_path
    llm = None
    
    # Initialize model with error handling
    try:
        if args.mode == "PLAIN":
            llm = PlainLLM(llm_modelid, args.quantize)
        elif args.mode == "SPARAGRAPH":
            llm = SpaRAGraph(llm_modelid, 
                             RDFGraphIndex(rdf_input_file, RDF_Composition(ApproximateCM())), 
                             SpacyNER(), 
                             ANNFewShot(),
                             args.quantize, 
                             args.few_shot)
        elif args.mode == "FAISS":
            llm = BaseRAG(llm_modelid, 
                             FAISSIndex(rdf_input_file),
                             args.k,
                             args.quantize)
    except Exception as e:
        print(bcolors.RED + f"Failed to initialize model: {e}" + bcolors.ENDC)
        raise
        
    # Load queries
    df = pd.read_csv(args.query_dataset_path)
    output_path = args.query_result_path

    # Write header
    col_num = len(df.columns)
    with open(output_path, "w") as f:
        if col_num == 3:
            f.write("type,query,truth,response\n")
        elif col_num == 2:
            f.write("query,truth,response\n")
        else:
            print(bcolors.RED + f"Invalid number of columns in file: {col_num}" + bcolors.ENDC)

    start = time.time()
    failed_queries = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating queries..."):
        if col_num == 3:
            qtype = row['type']
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
        else:
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
            qtype = args.qtype

            
        if index >= 0:
            # Periodic cleanup (every 5 queries)
            if index % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
            # More aggressive cleanup every 20 queries
            if index % 20 == 0 and index > 0:
                print(bcolors.CYAN + f"\n[Checkpoint {index}] Performing memory cleanup..." + bcolors.ENDC)
                torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            print(bcolors.BLUE + f"Query Type: {qtype}, Query: {query}" + bcolors.ENDC)

            try:
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
                
                if response == "no response":
                    failed_queries += 1

            except Exception as e:
                print(bcolors.RED + f"Unexpected error processing query: {e}" + bcolors.ENDC)
                response = "no response"
                failed_queries += 1

            if isinstance(response, list):
                response = "; ".join(response)

            print(bcolors.RED + "response: " + str(response) + bcolors.ENDC)

            # Write result
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

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"📊 Total queries: {len(df)}")
    print(f"❌ Failed queries: {failed_queries} ({failed_queries/len(df)*100:.1f}%)")
    print(f"{'='*60}")


if __name__=="__main__":
    main()
