from collections import Counter
from tqdm import tqdm
from llm_class import PlainLLM, SparagiRDF, REASONING
import torch
import argparse
import re
import pandas as pd
import time
torch.cuda.empty_cache()

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

FEW_SHOT_NUM = 0
# REASONING_TYPE = REASONING.INTERNAL
REASONING_TYPE = REASONING.EXTERNAL
llm = None

def main():
    global llm, FEW_SHOT_NUM
    parser = argparse.ArgumentParser(description="Run an LLM with optional RAG functionality.")
    parser.add_argument("-mode", type=str, default="PLAIN", help="Path of the query dataset to use")
    parser.add_argument("-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM model to load")
    parser.add_argument("-quantize", type=int, default=4, help="Bits to quantize the model to. Options: [4,8]")
    parser.add_argument("-few_shot", type=int, default=0, help="Few-shot examples to add during inference.")
    parser.add_argument("-rdf", type=str, default="datasets/CSZt.nt", help="The RDF dataset in .nt format.")
    args = parser.parse_args()

    llm_modelid = args.model
    FEW_SHOT_NUM = args.few_shot
    rdf_input_file = args.rdf
    
    if args.mode == "PLAIN":
        print(bcolors.GREEN + "Using " + args.model + bcolors.ENDC)
        llm = PlainLLM(llm_modelid, args.quantize)
        while True:
            user_input = input("Give a prompt (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            else:
                print(bcolors.BLUE + "Prompt:\n" + user_input + bcolors.ENDC )
                print(bcolors.RED + "Response:\n" + llm.chat(user_input) + bcolors.ENDC)
                
    elif args.mode == "SPARAGRAPH":    
        print(bcolors.GREEN + "Using SPARAGRAPH on " + args.model + bcolors.ENDC)
        llm = SparagiRDF(llm_modelid, rdf_input_file, args.quantize)
        while True:
            user_input = input("Give a prompt (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            else:
                print(bcolors.BLUE + "Prompt:\n" + user_input + bcolors.ENDC)
                print(bcolors.RED + "Response:\n" + llm.chat(user_input, REASONING_TYPE) + bcolors.ENDC)
    else:
        print(f"Error, Invalid mode: {args.mode}")
    

   
    



if __name__=="__main__":
    main()
