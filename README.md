# SpaRAGraph

Source code for 'SpaRAGraph: Spatial Reasoning using Retrieval-Augmented Generation'.

A framework that (i) performs spatial-to-RDF data processing to capture spatial relations between nearby entities (through SpaTex), (ii) indexes relation-RDFs using a graph to facilitate semantic traversal and (iii) retrieves the relevant context to a question at inference time, contextualizing it with factual, spatial information enhancing the LLM’s accuracy.

## Requirements

- spaCy : [GitHub repo](https://github.com/explosion/spacy-huggingface-hub)
- FAISS : [GitHub repo](https://github.com/facebookresearch/faiss)
- SentenceTransformers : [HF repo](https://huggingface.co/sentence-transformers)
- Python RDFLib

We also include a `requirements.txt` file with all the libraries and the versions we used in our experiments.

## Contents

- `datasets/CSZt.nt` : the RDF dataset comprising of the relations between Counties-States-Zipcodes in the U.S.A. Generated using 'SpaTex' ([GitHub repo](https://github.com/ThanGeo/SpaTex---Spatial-To-Text-data-toolkit)).

- `queries/` : the 3 query datasets (binary, multiclass and multilabel classification) of 'The Spatial Reasoning Benchmark' ([HF link](https://huggingface.co/datasets/Rammen/SpatialReasoning)), as found in the official [GitHub repo](https://github.com/ThanGeo/spatial-inference-benchmark).

- `chat.py` : a sample python code for chatting with a model with SpaRAGraph either enabled or disabled. Arguments:
- - `-mode` : [`PLAIN` (default), `SPARAGRAPH`]
- - `-model` : the model id to load.
- - `-rdf` : path to the RDF dataset to load (when SPARAGRAPH is enabled).
- - `-quantize` : [4,8] quantization for model
- - `-few_shot` : [1,2,3,4] few shot setting (static examples, tailored for CSZt dataset).

- `getResponses.py` : the code that was used to get responses to each query dataset from the models, including the instructions per-task. To run a dataset through SpaRAGraph, use:

```
python3 getResponsesLLM.py -model "modelID" -quantize 4  -query_dataset_path "queries/queries_yesno.csv" -qtype "yes/no" -query_result_path "output.csv"
```

- - `-qtype` must match with the query dataset given as input by the `-query_dataset_path` argument (in order to use the correct instruction). Types include: [`yes/no`, `radio`, `checkbox`].

## Notes 
In this preliminary work, the path search and context generation works for questions regarding specific entities. For open-ended questions (e.g. What are all the zipcodes inside the State of Florida?), the path search and context generation methods need to be adjusted so that ALL paths from the specified node are searched to zipcode nodes. This pertains to future work.

## Copyrights
MIT all rights reserved. 