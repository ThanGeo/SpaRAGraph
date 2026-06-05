
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from collections import Counter
from tqdm import tqdm
import torch
import argparse
import re
import yaml
import pandas as pd
import time
import gc

from index.rdf_graph_index import RDFGraphIndex
from index.faiss_index import FAISSIndex
from CG.rdf_composition import RDF_Composition
from NER.spacy_ner import SpacyNER
from CG.composition_matrix import ApproximateCM

from fewshot.ann_similar import ANNFewShot

from utils import bcolors, load_dotenv

load_dotenv()

from model_module import PlainLLM, SpaRAGraph, BaseRAG

# Clear cache at startup
if torch.cuda.is_available():
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
                tqdm.write(bcolors.RED + f"CUDA error (attempt {attempt+1}/{max_retries}): {str(e)}" + bcolors.ENDC)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

                if attempt < max_retries - 1:
                    tqdm.write(bcolors.WARNING + "Retrying after cleanup..." + bcolors.ENDC)
                    time.sleep(1)
                else:
                    tqdm.write(bcolors.RED + "Max retries reached, skipping query" + bcolors.ENDC)
                    raise
            else:
                raise

UNRECOGNIZED = "unrecognized"

def _warn_invalid(original, processed, valid_values):
    tqdm.write(
        bcolors.WARNING +
        f'  [!] Raw response  : "{original}"\n'
        f'      After cleanup : "{processed}" — not in [{", ".join(valid_values)}]\n'
        f'      Recorded as   : "{UNRECOGNIZED}"' +
        bcolors.ENDC
    )

def getYesNoResponse(query):
    last_raw = ""
    responses = []
    valid = ['yes', 'no']

    for i in range(REPEAT_FACTOR):
        try:
            raw = safe_cuda_operation(llm.runQuery, query, "BINARY")
            last_raw = raw
            cleaned = raw.replace("\n", " ").lower()
            cleaned = regex.sub('', cleaned)
            responses.append(cleaned)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            tqdm.write(bcolors.WARNING + f"  [!] CUDA error, skipping query ({str(e)[:80]})" + bcolors.ENDC)
            responses.append(UNRECOGNIZED)
            break

    most_prominent_response = Counter(responses).most_common(1)[0][0]

    if most_prominent_response not in valid:
        _warn_invalid(last_raw, most_prominent_response, valid)
        most_prominent_response = UNRECOGNIZED

    return most_prominent_response

def getRadioResponse(query):
    last_raw = ""
    responses = []
    valid = ['a', 'b', 'c', 'd', 'e']

    for i in range(REPEAT_FACTOR):
        try:
            raw = safe_cuda_operation(llm.runQuery, query, "MULTICLASS")
            last_raw = raw
            cleaned = raw.replace("\n", " ").lower().replace(".", "")
            cleaned = regex.sub('', cleaned)
            responses.append(cleaned)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            tqdm.write(bcolors.WARNING + f"  [!] CUDA error, skipping query ({str(e)[:80]})" + bcolors.ENDC)
            responses.append(UNRECOGNIZED)
            break

    most_prominent_response = Counter(responses).most_common(1)[0][0]

    if most_prominent_response not in valid:
        if most_prominent_response and most_prominent_response[0] in valid:
            most_prominent_response = most_prominent_response[0]
        else:
            _warn_invalid(last_raw, most_prominent_response, valid)
            most_prominent_response = UNRECOGNIZED

    return most_prominent_response

def getCheckboxResponse(query):
    last_raw = ""
    responses = []
    valid = ['a', 'b', 'c', 'd', 'e']

    for i in range(REPEAT_FACTOR):
        try:
            raw = safe_cuda_operation(llm.runQuery, query, "MULTILABEL")
            last_raw = raw
            cleaned = raw.replace("\n", " ").lower().replace(".", "").replace(" ", "")
            responses.append(cleaned)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            tqdm.write(bcolors.WARNING + f"  [!] CUDA error, skipping query ({str(e)[:80]})" + bcolors.ENDC)
            responses.append(UNRECOGNIZED)
            break

    most_prominent_response = Counter(responses).most_common(1)[0][0]

    for token in most_prominent_response.split(','):
        if token not in valid + ['']:
            _warn_invalid(last_raw, most_prominent_response, valid)
            most_prominent_response = UNRECOGNIZED
            break

    return most_prominent_response

def load_config(config_path, overrides):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides (only keys explicitly passed)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def main():
    global llm
    parser = argparse.ArgumentParser(description="Run an LLM with optional RAG functionality.")
    parser.add_argument("config", type=str, help="Path to a YAML config file (e.g. configs/sparagraph.yaml)")
    # Optional overrides — any config key can be overridden from the command line
    parser.add_argument("--mode", type=str, default=None, help="Override mode [PLAIN,SPARAGRAPH,FAISS]")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model id")
    parser.add_argument("--rdf_dataset", type=str, default=None, help="Override RDF dataset path")
    parser.add_argument("--query_dataset", type=str, default=None, help="Override query dataset path")
    parser.add_argument("--result_path", type=str, default=None, help="Override output CSV path")
    parser.add_argument("--qtype", type=str, default=None, help="Override query type [yes/no,radio,checkbox]")
    parser.add_argument("--quantize", type=int, default=None, help="Override quantization bits [4,8]")
    parser.add_argument("--few_shot", type=int, default=None, help="Override few-shot count")
    parser.add_argument("--k", type=int, default=None, help="Override k for FAISS retrieval")
    parser.add_argument("--device", type=str, default=None, help="Override device [auto, cpu]")
    parser.add_argument("--verbose", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="Override verbose output [true, false]")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace auth token (overrides HF_TOKEN env var)")
    raw = parser.parse_args()

    cfg = load_config(raw.config, {
        "mode": raw.mode,
        "model": raw.model,
        "rdf_dataset": raw.rdf_dataset,
        "query_dataset": raw.query_dataset,
        "result_path": raw.result_path,
        "qtype": raw.qtype,
        "quantize": raw.quantize,
        "few_shot": raw.few_shot,
        "k": raw.k,
        "device": raw.device,
        "verbose": raw.verbose,
        "hf_token": raw.hf_token,
    })

    W = 54
    mode_label = cfg["mode"]
    print(bcolors.CYAN + "─" * W)
    print(f" SpaRAGraph Benchmark Runner")
    print("─" * W + bcolors.ENDC)
    print(f"  Mode      : {bcolors.GREEN}{mode_label}{bcolors.ENDC}")
    print(f"  Model     : {cfg['model']}")
    if "rdf_dataset" in cfg:
        print(f"  RDF data  : {cfg['rdf_dataset']}")
    print(f"  Queries   : {cfg['query_dataset']}")
    print(f"  Query type: {cfg.get('qtype') or 'from CSV'}")
    print(f"  Output    : {cfg['result_path']}")
    print(f"  Quantize  : {cfg.get('quantize', 4)}-bit")
    if mode_label == "SPARAGRAPH":
        print(f"  Few-shot  : {cfg.get('few_shot', 0)}")
    if mode_label == "FAISS":
        print(f"  k         : {cfg.get('k', 1)}")
    device = cfg.get("device", "auto")
    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    print(f"  Device    : {device}")
    print(f"  HF token  : {'set' if hf_token else 'not set (public models only)'}")
    print(f"  Verbose   : {'on' if cfg.get('verbose', True) else 'off'}")
    print(bcolors.CYAN + "─" * W + bcolors.ENDC)

    llm_modelid = cfg["model"]
    rdf_input_file = cfg.get("rdf_dataset", "datasets/CSZt.nt")
    llm = None

    # Initialize model with error handling
    try:
        if cfg["mode"] == "PLAIN":
            llm = PlainLLM(llm_modelid, cfg.get("quantize", 4), device, hf_token)
        elif cfg["mode"] == "SPARAGRAPH":
            llm = SpaRAGraph(llm_modelid,
                             RDFGraphIndex(rdf_input_file, RDF_Composition(ApproximateCM())),
                             SpacyNER(),
                             ANNFewShot(),
                             cfg.get("quantize", 4),
                             cfg.get("few_shot", 0),
                             device,
                             hf_token)
        elif cfg["mode"] == "FAISS":
            llm = BaseRAG(llm_modelid,
                             FAISSIndex(rdf_input_file),
                             cfg.get("k", 1),
                             cfg.get("quantize", 4),
                             device,
                             hf_token)
    except Exception as e:
        print(bcolors.RED + f"Failed to initialize model: {e}" + bcolors.ENDC)
        raise

    # Load queries
    df = pd.read_csv(cfg["query_dataset"])
    output_path = cfg["result_path"]

    col_num = len(df.columns)
    if col_num == 2 and not cfg.get("qtype"):
        print(bcolors.RED + "  Error: query CSV has no 'type' column — 'qtype' must be set in the config." + bcolors.ENDC)
        raise SystemExit(1)

    if col_num not in (2, 3):
        print(bcolors.RED + f"Error: unexpected number of columns ({col_num}) in query CSV — expected 2 or 3." + bcolors.ENDC)
        raise SystemExit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write header
    with open(output_path, "w") as f:
        if col_num == 3:
            f.write("type,query,truth,response\n")
        else:
            f.write("query,truth,response\n")

    start = time.time()
    failed_queries = 0
    verbose = cfg.get("verbose", True)

    def _fmt_block(label, text, color):
        prefix = f"     {label}: "
        indent = " " * len(prefix)
        lines  = [l for l in text.strip().splitlines() if l.strip()]
        return color + prefix + ("\n" + indent).join(lines) + bcolors.ENDC

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating queries..."):
        if col_num == 3:
            qtype = row['type']
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
        else:
            query = row['query']
            truth = row['truth'] if 'truth' in df.columns else ""
            qtype = cfg.get("qtype")

        # Periodic CUDA cleanup
        if index % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        if index % 20 == 0 and index > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if verbose:
            flat_query = " ".join(query.split())  # collapse newlines (radio/checkbox options span lines)
            tqdm.write(bcolors.BLUE + f"[{index+1}] {(qtype or 'unknown'):<8} {flat_query}" + bcolors.ENDC)

        try:
            if qtype == "yes/no":
                response = getYesNoResponse(query)
            elif qtype == "radio":
                response = getRadioResponse(query)
            elif qtype == "checkbox":
                response = getCheckboxResponse(query)
            else:
                tqdm.write(bcolors.WARNING + f"  [!] Unknown query type '{qtype}'" + bcolors.ENDC)
                response = UNRECOGNIZED

            if response == UNRECOGNIZED:
                failed_queries += 1

        except Exception as e:
            tqdm.write(bcolors.RED + f"  [!] Unexpected error: {e}" + bcolors.ENDC)
            response = UNRECOGNIZED
            failed_queries += 1

        if isinstance(response, list):
            response = "; ".join(response)

        if verbose:
            context = getattr(llm, 'last_context', None)
            if context and context.strip():
                tqdm.write(_fmt_block("Context ", context, bcolors.PURPLE))

            prompt = getattr(llm, 'last_user_message', None)
            if prompt and prompt.strip():
                tqdm.write(_fmt_block("Prompt  ", prompt, bcolors.MAGENTA))

            color = bcolors.GREEN if response != UNRECOGNIZED else bcolors.RED
            tqdm.write(color + f"     → {response}" + bcolors.ENDC)

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
