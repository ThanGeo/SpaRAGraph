import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import gc

from utils import bcolors

from core.base_index import BaseIndex
from core.base_ner import BaseNER
from core.base_fewshot import BaseFewShot

# CRITICAL: Set these BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

# Base LLM class
class LLM:
    model = None
    terminators = None
    tokenizer = None

    def __init__(self, llm_modelid, quantize_bits=None, device="auto", hf_token=None):
        self.llm_modelid = llm_modelid
        self.device = device
        self.hf_token = hf_token

        if device == "cpu":
            if quantize_bits is not None:
                print(bcolors.WARNING + "Quantization is not supported on CPU — ignoring quantize setting." + bcolors.ENDC)
            self.bnb_config = None
        elif quantize_bits == 4:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantize_bits == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        else:
            self.bnb_config = None

        self.loadModel()

    def getSystemRole(self):
        return self.system_role
    
    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant."}]

    def loadModel(self):
        if self.device != "cpu":
            torch.cuda.empty_cache()
            gc.collect()

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_modelid, token=self.hf_token)

        load_kwargs = dict(
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=self.hf_token,
        )
        if self.bnb_config is not None:
            load_kwargs["quantization_config"] = self.bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(self.llm_modelid, **load_kwargs)
        
        # Set to eval mode for inference stability
        self.model.eval()
        
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
        if self.device != "cpu":
            torch.cuda.empty_cache()

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=4096,
                    do_sample=True,
                    eos_token_id=self.terminators,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            response = outputs[0][input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)

            del input_ids, outputs
            if self.device != "cpu":
                torch.cuda.empty_cache()

            return decoded_response

        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e):
                print(bcolors.RED + f"CUDA Error in generateAndDecode: {str(e)}" + bcolors.ENDC)
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                print(bcolors.WARNING + "Retrying after CUDA error..." + bcolors.ENDC)
                return self.generateAndDecode(messages)
            else:
                raise

    def generate(self, messages):
        messages = self.getSystemRole() + messages
        return self.generateAndDecode(messages)

# Plain LLM - No RAG
class PlainLLM(LLM):
    def answerQuery(self, prompt, query_type):
        messages = []
        if query_type == "yes/no":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"Question: {prompt}\nInstruction: Respond with only 'yes' or 'no'. Do not include any explanation or additional text."})
        elif query_type == "radio":
            self.generateSystemRole()
            messages.append({"role": "user", "content": f"{prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        elif query_type == "checkbox":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"{prompt}\nInstruction: Respond only with the letter(s) separated by commas (e.g., 'a,b,c'). If 'e' (none of the above) is selected, it must be the only option. Do not include any explanation or additional text."})

        self.last_user_message = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return super().generate(messages)

    def runQuery(self, query, qtype):
        mapping = {"BINARY": "yes/no", "MULTICLASS": "radio", "MULTILABEL": "checkbox"}
        return self.answerQuery(query, mapping.get(qtype, qtype))
    
    def chat(self, prompt, few_shot_num=0):
        messages = []
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant."}]
        messages.append({"role": "user", "content": prompt})
        self.last_user_message = prompt
        return super().generate(messages)
    
    def __init__(self, llm_modelid, quantize_bits=None, device="auto", hf_token=None):
        super().__init__(llm_modelid, quantize_bits, device, hf_token)
        self.generateSystemRole()

        print(bcolors.GREEN + f"Loaded model {llm_modelid}-plain: \
            \n - Quantization: {quantize_bits} bits \
            \n - Device: {device} \
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
                 few_shot_num=0,
                 device="auto",
                 hf_token=None):

        self.index_module = index_module
        self.ner_module = ner_module
        self.few_shot_num = few_shot_num
        self.fewshot_module = fewshot_module

        super().__init__(llm_modelid, quantize_bits, device, hf_token)
        self.generateSystemRole()

        print(bcolors.GREEN + f"Loaded model {llm_modelid}-SpaRAGraph: \
            \n - Quantization: {quantize_bits} bits \
            \n - Device: {device} \
            \n - {few_shot_num}-shot \
            \n - Dataset: {index_module.getDatasetPath()} \
            \n - Index type: {index_module.getType()} \
            \n - NER type: {ner_module.getType()} \
            " + bcolors.ENDC)

    def runQuery(self, query, qtype):
        referenced_entities = self.ner_module.extract_entities(query)
        context = self.index_module.generateContext(referenced_entities)
        self.last_context = context
        messages = []

        if self.few_shot_num > 0:
            messages += self.fewshot_module.getKshot(query, self.few_shot_num)

        if qtype == "BINARY":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"Context: {context} \nQuestion: {query}\nInstruction: Respond with only 'yes' or 'no'. Do not include any explanation or additional text."})
        elif qtype == "MULTICLASS":
            self.generateSystemRole()
            messages.append({"role": "user", "content": f"Context: {context} {query}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        elif qtype == "MULTILABEL":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"Context: {context} \n{query}\nInstruction: Respond only with the letter(s) separated by commas (e.g., 'a,b,c'). If 'e' (none of the above) is selected, it must be the only option. Do not include any explanation or additional text."})

        self.last_user_message = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return super().generate(messages)

    def chat(self, query):
        referenced_entities = self.ner_module.extract_entities(query)
        context = self.index_module.generateContext(referenced_entities)
        self.last_context = context
        messages = []

        if self.few_shot_num > 0:
            messages += self.fewshot_module.getKshot(query, self.few_shot_num)

        message = f"Context: {context}\n{query}" if context and context.strip() else query
        messages.append({"role": "user", "content": message})
        self.last_user_message = message
        return super().generate(messages)

# base RAG LLM
class BaseRAG(LLM):
    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately."}]

    def __init__(self, llm_modelid: str,
                 index_module: BaseIndex,
                 k,
                 quantize_bits=None,
                 device="auto",
                 hf_token=None):

        self.index_module = index_module
        self.k = k

        super().__init__(llm_modelid, quantize_bits, device, hf_token)
        self.generateSystemRole()

        print(bcolors.GREEN + f"Loaded model {llm_modelid}-BaseRAG: \
            \n - Quantization: {quantize_bits} bits \
            \n - Device: {device} \
            \n - Dataset: {index_module.getDatasetPath()} \
            \n - Index type: {index_module.getType()} \
            " + bcolors.ENDC)

    def runQuery(self, query, qtype):
        relevantTriplets = self.index_module.retrieveK(query, self.k)
        context = self.index_module.generateContext(relevantTriplets)
        messages = []

        if qtype == "BINARY":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"Context: {context} \nQuestion: {query}\nInstruction: Respond with only 'yes' or 'no'. Do not include any explanation or additional text."})
        elif qtype == "MULTICLASS":
            self.generateSystemRole()
            messages.append({"role": "user", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": f"{query}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        elif qtype == "MULTILABEL":
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            messages.append({"role": "user", "content": f"Context: {context} \n{query}\nInstruction: Respond only with the letter(s) separated by commas (e.g., 'a,b,c'). If 'e' (none of the above) is selected, it must be the only option. Do not include any explanation or additional text."})

        self.last_user_message = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return super().generate(messages)
