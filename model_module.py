import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rdflib import Graph, URIRef

from utils import bcolors

from core.base_index import BaseIndex
from core.base_ner import BaseNER
from core.base_fewshot import BaseFewShot

class REASONING:
    RDF_GRAPH_COMPOSITION = 0,
    INTERNAL = 1

# Base LLM class
class LLM:
    llm_modelid = ""
    # default bnb quantization config - none
    bnb_config = BitsAndBytesConfig()
    model = None
    terminators = None
    tokenizer = None
    system_role = []

    def __init__(self, llm_modelid, quantize_bits=None):
        # model id
        self.llm_modelid = llm_modelid
        # set quantization
        if quantize_bits == 4:
            self.bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantize_bits == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # bnb_4bit_quant_type="nf4",  # or "fp4"
                # bnb_4bit_compute_dtype=torch.float16,
                # bnb_4bit_use_double_quant=True
            )   
        else:
            self.bnb_config = BitsAndBytesConfig()
        # load model (TE and LLM)
        self.loadModel()

    def getSystemRole(self):
            return self.system_role
    
    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant."}]

    def loadModel(self):        
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_modelid)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_modelid,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.bnb_config
            )
        # terminators
        self.terminators = []
        if self.tokenizer.eos_token_id is not None:
            self.terminators.append(self.tokenizer.eos_token_id)
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None:
                self.terminators.append(eot_id)
        except Exception:
            pass

    def generateAndDecode(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=4096,
            do_sample=True,
            eos_token_id=self.terminators,
            temperature=0.7,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
        return decoded_response

    def generate(self, messages):
        # add the system role (doesnt work on all models)
        messages = self.getSystemRole() + messages
        return self.generateAndDecode(messages)

# Plain LLM - No RAG
class PlainLLM(LLM):
    def answerQuery(self, prompt, query_type):
        messages = []
        # set instruction in system based on query type
        if query_type == "yes/no":
            # put instruction in the system role
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            message = f"Question: {prompt}"
            messages.append({"role": "user", "content": f"{message}"})
        elif query_type == "radio":
            self.generateSystemRole()   # reset
            # for radio, do not override the system role. for some reason the instruction works best if paired with the question/options.
            messages.append({"role": "user", "content": f"{prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        
            # for mistral, a single user message must be given.
            # message = f"{prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."
            # messages.append({"role": "user", "content": f"{message}"})
        elif query_type == "checkbox":
            # put instruction in the system role
            # self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'None of the above' is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            message = f"{prompt}"
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"{message}"})
              
        # generate and return
        return super().generate(messages)
    
    def chat(self, prompt, few_shot_num): 
        messages = []        
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant."}]
        message = f"{prompt}"
        messages.append({"role": "user", "content": f"{message}"})
        
        # generate and return
        return super().generate(messages)
    
    def __init__(self, llm_modelid, quantize_bits=None):
        super().__init__(llm_modelid, quantize_bits)
        # set system role
        self.generateSystemRole()

        print(bcolors.GREEN + f"Loaded model {llm_modelid}-plain: \
            \n - Quantization: {quantize_bits} bits \
            " + bcolors.ENDC)

# SpaRAGraph LLM
class SpaRAGraph(LLM):
    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately."}]

    def __init__(self, llm_modelid: str, 
                 index_module: BaseIndex, 
                 ner_module: BaseNER,
                 fewshot_module: BaseFewShot,
                 quantize_bits=None, 
                 few_shot_num=0):
        
        # Index module
        self.index_module = index_module
        # NER module
        self.ner_module = ner_module
        # few-shot option
        self.few_shot_num = few_shot_num
        self.fewshot_module = fewshot_module
        # load model
        super().__init__(llm_modelid, quantize_bits)

        # set system role
        self.generateSystemRole()

        print(bcolors.GREEN + f"Loaded model {llm_modelid}-SpaRAGraph: \
            \n - Quantization: {quantize_bits} bits \
            \n - {few_shot_num}-shot \
            \n - Dataset: {index_module.getDatasetPath()} \
            \n - Index type: {index_module.getType()} \
            \n - NER type: {ner_module.getType()} \
            " + bcolors.ENDC)

    def runQuery(self, query, qtype):
        # get the referenced entities
        referenced_entities = self.ner_module.extract_entities(query)

        # generate the context through the index
        context = self.index_module.generateContext(referenced_entities)

        # the formatted message to the LLM
        messages = []

        # generate few shot examples
        if self.few_shot_num > 0:
            # add few shot examples
            messages += self.fewshot_module.getKshot(query, self.few_shot_num)
        
        # set instruction in system based on query type
        if qtype == "BINARY":
            # put instruction in the system role
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            message = f"Context: {context} \nQuestion: {query}"
            messages.append({"role": "user", "content": f"{message}"})
        elif qtype == "MULTICLASS":
            self.generateSystemRole()   # reset
            # for radio, do not override the system role. for some reason the instruction works best if paired with the question/options.
            messages.append({"role": "user", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": f"{query}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        
            # for mistral and geocode-gpt, a single user message must be given.
            # message = f"Context: {context} {prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."
            # messages.append({"role": "user", "content": f"{message}"})
        elif qtype == "MULTILABEL":
            # put instruction in the system role
            # self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'None of the above' is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            message = f"Context: {context} \n{query}"
            messages.append({"role": "user", "content": f"{message}"})
              
        # generate and return
        print(messages)
        return super().generate(messages)
