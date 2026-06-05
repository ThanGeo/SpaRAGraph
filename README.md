# SpaRAGraph

Source code for *[SpaRAGraph: Spatial Reasoning using Retrieval-Augmented Generation](https://dl.acm.org/doi/full/10.1145/3799424)*.

A framework that (i) processes spatial data into RDF triples capturing spatial relations between geographic entities, (ii) indexes those triples as a traversable graph, and (iii) retrieves relevant context at inference time by performing entity-grounded BFS path search, enriching the LLM prompt with factual spatial information.

## Requirements

- Python 3.9
- CUDA-capable GPU recommended (CPU inference is supported but slow; quantization requires GPU)

Key libraries (see `requirements.txt` for pinned versions):

| Library | Purpose |
|---|---|
| `torch`, `transformers`, `bitsandbytes` | LLM loading and quantization |
| `sentence-transformers`, `faiss-cpu` | Embedding-based entity grounding and retrieval |
| `spacy` | Named entity recognition |
| `rdflib` | RDF graph parsing and traversal |
| `pyyaml` | YAML config loading |

## Setup

```bash
python3.9 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Authentication

Gated models (Llama, Mistral, Gemma, etc.) require a HuggingFace account token. There are three ways to provide it, in order of precedence:

**1. `.env` file (recommended)** — copy `env.example` to `.env` and fill in your token. It is loaded automatically at startup and is gitignored.

```bash
cp env.example .env
# edit .env and set HF_TOKEN=hf_your_token_here
```

**2. Environment variable** — set `HF_TOKEN` in your shell (or CI environment):

```bash
export HF_TOKEN=hf_your_token_here
```

**3. Config file** — add `hf_token: hf_your_token_here` to any YAML config. Only do this for local-only configs that you are sure will not be committed.

Public models (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) work without a token.

## Project structure

```
SpaRAGraph/
├── core/               # Abstract base classes (BaseIndex, BaseNER, BaseCG, BaseCM, BaseFewShot)
├── index/              # RDFGraphIndex (graph traversal) and FAISSIndex (flat retrieval)
├── CG/                 # Context generators: RDF_Composition + ApproximateCM (composition table)
├── NER/                # SpacyNER — extracts zipcodes, counties, states from query text
├── fewshot/            # ANNFewShot — retrieves similar examples via FAISS
├── datasets/           # CSZt.nt — County/State/Zipcode RDF dataset (generated with SpaTex)
├── queries/            # Query CSV files from the Spatial Reasoning Benchmark
│   └── queries_yesno_1.csv  # Single-query test file for quick validation
├── configs/            # YAML config files (one per run mode)
├── model_module.py     # LLM classes: PlainLLM, SpaRAGraph, BaseRAG
├── runBenchmark.py     # Batch benchmark entry point
└── chat.py             # Interactive chat entry point
```

The dataset (`datasets/CSZt.nt`) was generated with [SpaTex](https://github.com/ThanGeo/SpaTex---Spatial-To-Text-data-toolkit). Query files are from [The Spatial Reasoning Benchmark](https://huggingface.co/datasets/Rammen/SpatialReasoning) ([GitHub](https://github.com/ThanGeo/spatial-inference-benchmark)).

## Usage

All runs go through `runBenchmark.py` with a YAML config file:

```bash
python runBenchmark.py configs/<config>.yaml
```

Any config key can be overridden from the command line:

```bash
python runBenchmark.py configs/sparagraph.yaml --few_shot=3 --result_path=results/run1.csv
```

### Config reference

Both scripts share the same config format. Keys marked **benchmark** are used only by `runBenchmark.py` and ignored by `chat.py`.

| Key | Description | Values | Used by |
|---|---|---|---|
| `mode` | Run mode | `PLAIN`, `SPARAGRAPH`, `FAISS` | both |
| `model` | HuggingFace model ID | e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct` | both |
| `rdf_dataset` | Path to `.nt` RDF file | e.g. `datasets/CSZt.nt` | both (SPARAGRAPH) |
| `quantize` | Quantization bits (GPU only, ignored on CPU) | `4`, `8` | both |
| `few_shot` | Few-shot examples prepended to prompt | integer, default `0` | both (SPARAGRAPH) |
| `device` | Inference device | `auto` (GPU if available), `cpu` | both |
| `verbose` | Print per-query / per-message details | `true`, `false` | both |
| `query_dataset` | Path to query CSV | e.g. `queries/queries_yesno.csv` | **benchmark** |
| `result_path` | Output CSV path | e.g. `results/out.csv` | **benchmark** |
| `qtype` | Query type (required when CSV has no `type` column) | `yes/no`, `radio`, `checkbox` | **benchmark** |
| `k` | Triplets retrieved per query | integer, default `1` | **benchmark** (FAISS) |
| `hf_token` | HuggingFace auth token (prefer `.env` over this) | `hf_...` | both |

Query CSVs must have a `query` column and a `truth` column. If the CSV has a `type` column (one of `yes/no`, `radio`, `checkbox` per row), `qtype` in the config is not needed.

## Example runs

The following examples use `queries/queries_yesno_1.csv`, a single-query test file for quick validation:

```
query: "Is Zipcode 14610 adjacent to and southeast of Zipcode 14445?"
truth: no
```

**SpaRAGraph** — graph-traversal RAG (main method):
```bash
python runBenchmark.py configs/sparagraph.yaml
```

**Plain** — no retrieval, LLM only:
```bash
python runBenchmark.py configs/plain.yaml
```

**FAISS** — flat embedding-based retrieval baseline:
```bash
python runBenchmark.py configs/faiss.yaml
```

**CPU inference** (no GPU required, slower):
```bash
python runBenchmark.py configs/sparagraph.yaml --device=cpu
```

**Quiet run** (no per-query output, results in CSV only):
```bash
python runBenchmark.py configs/sparagraph.yaml --verbose=false
```

## Interactive chat

`chat.py` opens a free-form chat session. It supports `PLAIN` and `SPARAGRAPH` modes (`FAISS` is not supported). In SPARAGRAPH mode the graph is queried for context on every message, which is printed before the response when `verbose: true`.

`chat.py` accepts the same config files as `runBenchmark.py` — benchmark-only keys (`query_dataset`, `result_path`, `qtype`, `k`) are silently ignored:

```bash
python chat.py configs/sparagraph.yaml   # reuse benchmark config
python chat.py configs/plain.yaml
```

Minimal chat-only configs (no benchmark keys) are also provided:

```bash
python chat.py configs/chat_sparagraph.yaml
python chat.py configs/chat_plain.yaml
```

Example session (SPARAGRAPH, `verbose: true`):

```
You: Is New York County north of Kings County?
  Context  : New York County is north of Kings County New York.
Assistant: Yes, New York County (Manhattan) is located north of Kings County (Brooklyn).

You: exit
Goodbye.
```

The same CLI overrides apply (`--model`, `--device`, `--quantize`, `--few_shot`, `--rdf_dataset`).

## Notes

- Path search and context generation work for questions about specific named entities. Open-ended questions (e.g. *"What are all the zipcodes inside Florida?"*) require full-graph traversal from a node, which is left for future work.
- GPU with CUDA is strongly recommended for practical runtimes. When `device: cpu` is set, quantization is automatically disabled and `float32` is used.

## Cite this work

```bibtex
@article{georgiadis2026sparagraph,
  title={SpaRAGraph: Spatial reasoning using retrieval-augmented generation},
  author={Georgiadis, Thanasis and Pavlopoulos, John and Mamoulis, Nikos},
  journal={ACM Transactions on Spatial Algorithms and Systems},
  volume={12},
  number={3},
  pages={1--31},
  year={2026},
  publisher={ACM New York, NY}
}
```

## License

MIT
