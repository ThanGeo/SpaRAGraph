#!/usr/bin/env python3
"""
SpaRAGraph test suite — verifies setup and pipeline correctness.

Usage:
    python test.py                              # component tests only (fast, no model needed)
    python test.py --inference                  # also run end-to-end inference tests (slow)
    python test.py --inference --device cpu     # force CPU for inference tests
    python test.py --inference --model <id>     # use a specific HF model id
"""

import sys
import os
import re
import argparse
import subprocess
import tempfile
import shutil
import yaml

# ─── Terminal colours ────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; END = "\033[0m"
P = f"{GREEN}✓{END}"; F = f"{RED}✗{END}"; S = f"{YELLOW}─{END}"

# ─── Minimal test runner ─────────────────────────────────────────────────────
_results = []

def run(label, fn):
    try:
        fn()
        _results.append(("pass", label))
        print(f"  {P}  {label}")
        return True
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        _results.append(("fail", label, msg))
        print(f"  {F}  {label}  — {msg}")
        return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _results.append(("fail", label, msg))
        print(f"  {F}  {label}  — {msg}")
        return False

def skip(label, reason=""):
    _results.append(("skip", label))
    print(f"  {S}  {label}" + (f"  [{reason}]" if reason else ""))

def section(title):
    print(f"\n  {CYAN}{title}{END}\n  {'─' * len(title)}")

def summary():
    passed  = sum(1 for r in _results if r[0] == "pass")
    failed  = sum(1 for r in _results if r[0] == "fail")
    skipped = sum(1 for r in _results if r[0] == "skip")
    total   = len(_results)
    print(f"\n{'─' * 52}")
    parts = [f"{passed}/{total} passed"]
    if skipped: parts.append(f"{YELLOW}{skipped} skipped{END}")
    if failed:  parts.append(f"{RED}{failed} failed{END}")
    print("  " + "  |  ".join(parts))
    print(f"{'─' * 52}")
    return failed == 0


# ─── Response-parsing helpers (mirrors runBenchmark.py logic) ────────────────
_rx           = re.compile('[^a-zA-Z]')
_UNRECOGNIZED = "unrecognized"
_VALID        = {'a', 'b', 'c', 'd', 'e'}

def _parse_yesno(raw):
    c = _rx.sub('', raw.replace("\n", " ").lower())
    return c if c in ('yes', 'no') else _UNRECOGNIZED

def _parse_radio(raw):
    c = _rx.sub('', raw.replace("\n", " ").lower().replace(".", ""))
    if c in _VALID:        return c
    if c and c[0] in _VALID: return c[0]
    return _UNRECOGNIZED

def _parse_checkbox(raw):
    c = raw.replace("\n", " ").lower().replace(".", "").replace(" ", "")
    for tok in c.split(','):
        if tok not in _VALID | {''}:
            return _UNRECOGNIZED
    return c


# ─── 1. Environment ──────────────────────────────────────────────────────────
def test_environment():
    section("1. Environment")

    def _imports():
        import torch, transformers, bitsandbytes
        import yaml, pandas, rdflib, spacy
        import sentence_transformers, faiss, tqdm

    def _spacy_model():
        import spacy
        nlp = spacy.load("en_core_web_sm")
        assert nlp is not None

    def _dataset():
        assert os.path.isfile("datasets/CSZt.nt"), "datasets/CSZt.nt not found"

    def _query_files():
        for f in ["queries/queries_yesno_1.csv",
                  "queries/queries_radio_1.csv",
                  "queries/queries_checkbox_1.csv"]:
            assert os.path.isfile(f), f"{f} not found"

    def _configs():
        for f in ["configs/plain.yaml", "configs/sparagraph.yaml", "configs/faiss.yaml"]:
            assert os.path.isfile(f), f"{f} not found"
            with open(f) as fh:
                data = yaml.safe_load(fh)
            assert "mode" in data and "model" in data, f"{f} missing required keys"

    run("required packages importable",       _imports)
    run("spacy en_core_web_sm available",     _spacy_model)
    run("RDF dataset exists",                 _dataset)
    run("single-query test files exist",      _query_files)
    run("config files valid",                 _configs)


# ─── 2. RAG Pipeline ─────────────────────────────────────────────────────────
def test_pipeline():
    section("2. RAG Pipeline")

    YESNO    = "Is Zipcode 14610 adjacent to and southeast of Zipcode 14445?"
    RADIO    = ("Question: Select exactly one option (a-e) that best describes the relationship"
                " of Douglas County Washington in relation to Zipcode 98848 in terms of geography."
                " Options: a. contains b. adjacent to and northwest of"
                " c. adjacent to and north of d. southeast of e. none of the above")
    CHECKBOX = ("Question: Select all options that are west of Southampton County Virginia."
                " You may choose one or more options."
                " Options: a. Zipcode 23884 b. Zipcode 23882"
                " c. Zipcode 27979 d. Zipcode 23874 e. None of the above")

    _ner = _rdf = _faiss = _fewshot = None

    def _load_ner():
        nonlocal _ner
        from NER.spacy_ner import SpacyNER
        _ner = SpacyNER()

    def _ner_entities(query):
        entities = _ner.extract_entities(query)
        assert "ordered" in entities and "categorized" in entities
        flat = entities["ordered"]
        assert len(flat) >= 1, f"no entities extracted — got: {entities}"

    def _load_rdf():
        nonlocal _rdf
        from index.rdf_graph_index import RDFGraphIndex
        from CG.rdf_composition import RDF_Composition
        from CG.composition_matrix import ApproximateCM
        _rdf = RDFGraphIndex("datasets/CSZt.nt", RDF_Composition(ApproximateCM()))

    def _rdf_context():
        entities = _ner.extract_entities(YESNO)
        ctx = _rdf.generateContext(entities)
        assert ctx is None or isinstance(ctx, str)

    def _load_faiss():
        nonlocal _faiss
        from index.faiss_index import FAISSIndex
        _faiss = FAISSIndex("datasets/CSZt.nt")

    def _faiss_retrieve():
        results = _faiss.retrieveK(YESNO, k=3)
        assert isinstance(results, list) and len(results) > 0

    def _load_fewshot():
        nonlocal _fewshot
        from fewshot.ann_similar import ANNFewShot
        _fewshot = ANNFewShot()

    def _fewshot_retrieve():
        shots = _fewshot.getKshot(YESNO, k=2)
        assert isinstance(shots, list) and len(shots) > 0

    def _composition_empty():
        from CG.composition_matrix import ApproximateCM
        cm = ApproximateCM()
        assert cm.getCombinedRelation([]) == ""

    run("SpacyNER: initialise",                   _load_ner)
    run("SpacyNER: extract entities (yes/no)",    lambda: _ner_entities(YESNO))
    run("SpacyNER: extract entities (radio)",     lambda: _ner_entities(RADIO))
    run("SpacyNER: extract entities (checkbox)",  lambda: _ner_entities(CHECKBOX))
    run("RDFGraphIndex: load dataset",            _load_rdf)
    run("RDFGraphIndex: generate context",        _rdf_context)
    run("FAISSIndex: load and index dataset",     _load_faiss)
    run("FAISSIndex: retrieve top-k",             _faiss_retrieve)
    run("ANNFewShot: initialise",                 _load_fewshot)
    run("ANNFewShot: retrieve k-shot examples",   _fewshot_retrieve)
    run("ApproximateCM: composition lookup",      _composition_empty)


# ─── 3. Response Parsing ─────────────────────────────────────────────────────
def test_response_parsing():
    section("3. Response Parsing")

    def _yesno():
        assert _parse_yesno("yes")             == "yes"
        assert _parse_yesno("Yes")             == "yes"
        assert _parse_yesno("NO.")             == "no"
        assert _parse_yesno("no")              == "no"
        assert _parse_yesno("Yes, definitely") == _UNRECOGNIZED
        assert _parse_yesno("I don't know")    == _UNRECOGNIZED
        assert _parse_yesno("")                == _UNRECOGNIZED

    def _radio():
        assert _parse_radio("a")              == "a"
        assert _parse_radio("B")              == "b"
        assert _parse_radio("a.")             == "a"
        assert _parse_radio("a the answer")   == "a"
        assert _parse_radio("z")              == _UNRECOGNIZED
        assert _parse_radio("")               == _UNRECOGNIZED

    def _checkbox():
        assert _parse_checkbox("a,b")   == "a,b"
        assert _parse_checkbox("e")     == "e"
        assert _parse_checkbox("a,b,c") == "a,b,c"
        assert _parse_checkbox("a,z")   == _UNRECOGNIZED
        assert _parse_checkbox("x")     == _UNRECOGNIZED

    run("yes/no response parsing",   _yesno)
    run("radio response parsing",    _radio)
    run("checkbox response parsing", _checkbox)


# ─── 4. Inference (end-to-end) ───────────────────────────────────────────────
def test_inference(model, device, hf_token):
    section("4. Inference  (end-to-end, slow)")
    print(f"     model  : {model}")
    print(f"     device : {device}")
    print()

    outdir = tempfile.mkdtemp(prefix="sparagraph_test_")
    try:
        _run_all(model, device, hf_token, outdir)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def _run_all(model, device, hf_token, outdir):
    import pandas as pd

    def benchmark(mode, qtype, query_file, extra=None):
        """Write a temp config, run runBenchmark.py, return the output DataFrame."""
        outfile = os.path.join(outdir, f"{mode}_{qtype}.csv")
        cfg = {
            "mode":          mode,
            "model":         model,
            "query_dataset": query_file,
            "result_path":   outfile,
            "qtype":         qtype,
            "device":        device,
            "quantize":      None,
            "verbose":       False,
        }
        if mode in ("SPARAGRAPH", "FAISS"):
            cfg["rdf_dataset"] = "datasets/CSZt.nt"
        if mode == "SPARAGRAPH":
            cfg["few_shot"] = 0
        if mode == "FAISS":
            cfg["k"] = 1
        if hf_token:
            cfg["hf_token"] = hf_token
        if extra:
            cfg.update(extra)

        cfg_path = os.path.join(outdir, f"{mode}_{qtype}.yaml")
        with open(cfg_path, "w") as fh:
            yaml.dump(cfg, fh)

        result = subprocess.run(
            [sys.executable, "runBenchmark.py", cfg_path],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, (
            f"runBenchmark.py exited with code {result.returncode}\n"
            f"--- stdout (last 500 chars) ---\n{result.stdout[-500:]}\n"
            f"--- stderr (last 500 chars) ---\n{result.stderr[-500:]}"
        )
        assert os.path.isfile(outfile), "output CSV was not created"
        df = pd.read_csv(outfile)
        assert len(df) >= 1,             "output CSV is empty"
        assert "response" in df.columns, "output CSV missing 'response' column"
        return df

    def _valid_yesno(resp):
        assert resp in ("yes", "no", _UNRECOGNIZED), f"unexpected response: {resp!r}"

    def _valid_radio(resp):
        assert resp in list("abcde") + [_UNRECOGNIZED], f"unexpected response: {resp!r}"

    def _valid_checkbox(resp):
        tokens = str(resp).split(",")
        assert all(t in list("abcde") + [_UNRECOGNIZED] for t in tokens), \
            f"unexpected response: {resp!r}"

    run("PLAIN     — yes/no",   lambda: _valid_yesno(
        benchmark("PLAIN", "yes/no",   "queries/queries_yesno_1.csv").iloc[0]["response"]))
    run("PLAIN     — radio",    lambda: _valid_radio(
        benchmark("PLAIN", "radio",    "queries/queries_radio_1.csv").iloc[0]["response"]))
    run("PLAIN     — checkbox", lambda: _valid_checkbox(
        benchmark("PLAIN", "checkbox", "queries/queries_checkbox_1.csv").iloc[0]["response"]))
    run("SPARAGRAPH — yes/no",  lambda: _valid_yesno(
        benchmark("SPARAGRAPH", "yes/no", "queries/queries_yesno_1.csv").iloc[0]["response"]))
    run("FAISS     — yes/no",   lambda: _valid_yesno(
        benchmark("FAISS", "yes/no",   "queries/queries_yesno_1.csv").iloc[0]["response"]))


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SpaRAGraph test suite")
    parser.add_argument("--inference", action="store_true",
                        help="Also run end-to-end inference tests (requires a model)")
    parser.add_argument("--model",  default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HF model id to use for inference tests (default: TinyLlama)")
    parser.add_argument("--device", default="auto",
                        help="Device for inference tests: auto or cpu (default: auto)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for gated models (overrides HF_TOKEN env var)")
    args = parser.parse_args()

    # Load .env so HF_TOKEN is available if set
    try:
        from utils import load_dotenv
        load_dotenv()
    except Exception:
        pass

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    W = 52
    print(f"\n{CYAN}{'─' * W}")
    print(f" SpaRAGraph Test Suite")
    print(f"{'─' * W}{END}")

    test_environment()
    test_pipeline()
    test_response_parsing()

    if args.inference:
        test_inference(args.model, args.device, hf_token)
    else:
        section("4. Inference  (end-to-end, slow)")
        skip("inference tests", "run with --inference to enable")

    ok = summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
