import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import argparse
import yaml
import gc

from index.rdf_graph_index import RDFGraphIndex
from CG.rdf_composition import RDF_Composition
from NER.spacy_ner import SpacyNER
from CG.composition_matrix import ApproximateCM
from fewshot.ann_similar import ANNFewShot
from utils import bcolors, load_dotenv

load_dotenv()
from model_module import PlainLLM, SpaRAGraph

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


def load_config(config_path, overrides):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _print_context(context):
    prefix = "  Context  : "
    indent = " " * len(prefix)
    lines  = [l for l in context.strip().splitlines() if l.strip()]
    print(bcolors.PURPLE + prefix + ("\n" + indent).join(lines) + bcolors.ENDC)


def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM, optionally augmented with SpaRAGraph.")
    parser.add_argument("config", type=str, help="Path to a YAML config file (e.g. configs/chat_sparagraph.yaml)")
    parser.add_argument("--mode", type=str, default=None, help="Override mode [PLAIN, SPARAGRAPH]")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model id")
    parser.add_argument("--rdf_dataset", type=str, default=None, help="Override RDF dataset path")
    parser.add_argument("--quantize", type=int, default=None, help="Override quantization bits [4, 8]")
    parser.add_argument("--few_shot", type=int, default=None, help="Override few-shot count")
    parser.add_argument("--device", type=str, default=None, help="Override device [auto, cpu]")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace auth token (overrides HF_TOKEN env var)")
    raw = parser.parse_args()

    cfg = load_config(raw.config, {
        "mode": raw.mode,
        "model": raw.model,
        "rdf_dataset": raw.rdf_dataset,
        "quantize": raw.quantize,
        "few_shot": raw.few_shot,
        "device": raw.device,
        "hf_token": raw.hf_token,
    })

    mode_label = cfg["mode"]
    verbose     = cfg.get("verbose", True)
    device      = cfg.get("device", "auto")
    hf_token    = cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    W = 54
    print(bcolors.CYAN + "─" * W)
    print(f" SpaRAGraph Chat")
    print("─" * W + bcolors.ENDC)
    print(f"  Mode      : {bcolors.GREEN}{mode_label}{bcolors.ENDC}")
    print(f"  Model     : {cfg['model']}")
    if mode_label == "SPARAGRAPH":
        print(f"  RDF data  : {cfg.get('rdf_dataset', 'datasets/CSZt.nt')}")
        print(f"  Few-shot  : {cfg.get('few_shot', 0)}")
    print(f"  Quantize  : {cfg.get('quantize', 4)}-bit")
    print(f"  Device    : {device}")
    print(f"  HF token  : {'set' if hf_token else 'not set (public models only)'}")
    print(f"  Verbose   : {'on' if verbose else 'off'}")
    print(bcolors.CYAN + "─" * W + bcolors.ENDC)
    print()

    try:
        if mode_label == "PLAIN":
            llm = PlainLLM(cfg["model"], cfg.get("quantize", 4), device, hf_token)
        elif mode_label == "SPARAGRAPH":
            rdf_input_file = cfg.get("rdf_dataset", "datasets/CSZt.nt")
            llm = SpaRAGraph(cfg["model"],
                             RDFGraphIndex(rdf_input_file, RDF_Composition(ApproximateCM())),
                             SpacyNER(),
                             ANNFewShot(),
                             cfg.get("quantize", 4),
                             cfg.get("few_shot", 0),
                             device,
                             hf_token)
        else:
            print(bcolors.RED + f"Unknown mode '{mode_label}'. Valid options: PLAIN, SPARAGRAPH." + bcolors.ENDC)
            return
    except Exception as e:
        print(bcolors.RED + f"Failed to initialize model: {e}" + bcolors.ENDC)
        raise

    print(bcolors.CYAN + "Type your message below. Enter 'exit' or 'quit' to stop." + bcolors.ENDC)
    print()

    while True:
        try:
            user_input = input(bcolors.GREEN + "You: " + bcolors.ENDC).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            response = llm.chat(user_input)

            if verbose and mode_label == "SPARAGRAPH":
                context = getattr(llm, 'last_context', None)
                if context and context.strip():
                    _print_context(context)

            print(bcolors.BLUE + f"Assistant: {response}" + bcolors.ENDC)
            print()

        except Exception as e:
            print(bcolors.RED + f"[!] Error: {e}" + bcolors.ENDC)
            print()


if __name__ == "__main__":
    main()
