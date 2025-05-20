import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from geo_extractor import GeoEntityExtractor
from rdflib import Graph, URIRef
from rdf_graph_class import RDFGraphIndexer

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    PURPLE = '\033[35m'
    MAGENTA = '\033[35m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

# Base LLM class
class LLM:
    llm_modelid = ""
    # default bnb quantization config - none
    bnb_config = BitsAndBytesConfig()
    model = None
    terminators = None
    tokenizer = None
    system_role = []

    def getSystemRole(self):
            return self.system_role
    
    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a bot that answers spatial reasoning questions."}]

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
        # add the system role
        messages = self.getSystemRole() + messages
        # print(messages)
        return self.generateAndDecode(messages)

    # def __init__(self, llm_modelid):
    #     self.llm_modelid = llm_modelid
    #     self.loadModel()
    #     self.generateSystemRole()

# Plain LLM
class PlainLLM(LLM):
    def generate(self, prompt, k=0):
        messages = [{"role": "user", "content": "Question: " + prompt+"\n"}]
        return super().generate(messages)
    
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
        # set system role
        self.generateSystemRole()

# Spatial RAG LLM
class SparagiRDF(LLM):
    # graph
    graph_indexer = None
    # NER
    entity_extractor = None

    def generateSystemRole(self):
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately."}]

    def extractEntities(self, prompt):
        doc = self.ner(prompt)
        matches = self.matcher(doc)
        entities = {"zipcode": [], "state": [], "county": []}

        for match_id, start, end in matches:
            label = self.ner.vocab.strings[match_id]
            span = doc[start:end]

            if label == "ZIPCODE":
                entities["zipcode"].append(span.text)
            elif label == "STATE":
                # Extract the state name (skip "The State of")
                state_name = span[-1].text  # Last token is the state (e.g., "California" or "CA")
                entities["state"].append(state_name)
            elif label == "COUNTY":
                # Remove the suffix (e.g., "County") for cleaner output
                county_name = doc[start:end-1].text  # Exclude the last token (suffix)
                entities["county"].append(county_name)
        return entities

    def generateContext(self, start, end, combined_relation):
        # start = self.graph_indexer.get_local_name(start)
        # end = self.graph_indexer.get_local_name(end)       
        # combined_relation = self.graph_indexer.getCombinedRelation(path)
        context = ""
        if "contains" in combined_relation or "intersects with" in combined_relation:
            context += f"{start} {combined_relation} {end}.\n"
        else:
            context += f"{start} is {combined_relation} of {end}.\n"
            
        return context
    
    # NOTE: this works best for radio queries for some reason... (maybe system role?)
    # def generate(self, prompt):
    #     PROMPT = prompt
        
    #     # get entities
    #     entities = self.entity_extractor.extract_entities(prompt)
    #     # find start/end points in graph
    #     pairs = self.graph_indexer.find_start_end_pairs(entities)
    #     # get paths
    #     context = ""
    #     for start, end in pairs:
    #         path = self.graph_indexer.find_path_bfs_uri(start, end)
    #         # print(f"start: {start}")
    #         # print(f"end: {end}")
    #         # self.graph_indexer.printPath(path)
    #         if path:    
    #             clean_start = self.graph_indexer.get_local_name(start)
    #             clean_end = self.graph_indexer.get_local_name(end)       
    #             combined_relation = self.graph_indexer.getCombinedRelation(path)
    #             context += self.generateContext(clean_start, clean_end, combined_relation)
    #         else:
    #             # try looking in reverse
    #             path = self.graph_indexer.find_path_bfs_uri(end, start)
    #             # print(f"start: {end}")
    #             # print(f"end: {start}")
    #             # self.graph_indexer.printPath(path)
    #             if path:    
    #                 clean_start = self.graph_indexer.get_local_name(end)
    #                 clean_end = self.graph_indexer.get_local_name(start)       
    #                 combined_relation = self.graph_indexer.getInverseRelation(self.graph_indexer.getCombinedRelation(path))
    #                 context += self.generateContext(clean_start, clean_end, combined_relation)
        
    #     messages = []
    #     # append context (if any)
    #     if context != "":
    #         # PROMPT += f" \"Context: {context}\""
    #         messages.append({"role": "user", "content": f"Context:\n{context}"})

    #     # create formatted prompt for llm
    #     # messages = [{"role": "user", "content" : PROMPT}]
    #     messages.append({"role": "user", "content": f"Question:\n{prompt}"})
        
    #     # generate and return
    #     return super().generate(messages)
    
    def getFewShotMessages(self, ):
        pass
    
    def generate(self, prompt, query_type, few_shot_flag):
        PROMPT = prompt
        message_text = ""
        # get entities
        entities = self.entity_extractor.extract_entities(prompt)
        # find start/end points in graph
        pairs = self.graph_indexer.find_start_end_pairs(entities)
        # get paths
        context = ""
        for start, end in pairs:
            path = self.graph_indexer.find_path_bfs_uri(start, end)
            # if not path:
            #     print(f"None path for {start}->{end}")
            # print(f"start: {start}")
            # print(f"end: {end}")
            # self.graph_indexer.printPath(path)
            if path:    
                clean_start = self.graph_indexer.get_local_name(start)
                clean_end = self.graph_indexer.get_local_name(end)       
                combined_relation = self.graph_indexer.getCombinedRelation(path)
                context += self.generateContext(clean_start, clean_end, combined_relation)
            else:
                # try looking in reverse
                path = self.graph_indexer.find_path_bfs_uri(end, start)
                # if not path:
                #     print(f"None path for {start}->{end}")
                # print("Inverse...")
                # print(f"start: {end}")
                # print(f"end: {start}")
                # self.graph_indexer.printPath(path)
                if path:    
                    clean_start = self.graph_indexer.get_local_name(end)
                    clean_end = self.graph_indexer.get_local_name(start)       
                    combined_relation = self.graph_indexer.getInverseRelation(self.graph_indexer.getCombinedRelation(path))
                    context += self.generateContext(clean_start, clean_end, combined_relation)
        
        if few_shot_flag:
            # todo: implement few shot learning. For example: "Question: Is A east of B? Context: A is west of B." Generate something like:
            # messages = [{"role": "user", "content": "Context: X is west of Y.\nQuestion: Is X east of B?\nAnswer with only 'yes' or 'no'."},
            #             {"role": "assistant", "content": "no"},]
            pass
        
        messages = []
        # set instruction in system based on query type
        if query_type == "yes/no":
            # put instruction in the system role
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. Answer with only 'yes' or 'no'. Do not include any explanation or additional text."}]
            message = f"Context: {context} \nQuestion: {prompt}"
            messages.append({"role": "user", "content": f"{message}"})
        elif query_type == "radio":
            self.generateSystemRole()   # reset
            # for radio, do not override the system role. for some reason the instruction works best if paired with the question/options.
            messages.append({"role": "user", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": f"{prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."})
        elif query_type == "checkbox":
            # put instruction in the system role
            # self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'None of the above' is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately. If 'e' (none of the above) is selected, it must be the only selected option. Respond only with the letter(s) of the selected options, separated by commas (e.g., 'a,b,c'). Do not include any explanation or additional text."}]
            message = f"Context: {context} \n{prompt}"
            messages.append({"role": "user", "content": f"{message}"})
              
        # generate and return
        return super().generate(messages)

    def getPrompt(self, prompt, k):
        # retrieve documents
        # messages, retrievedDocumentIDs = self.RetrievalIndex.useIndex(prompt, k)
        # return messages[0]['content']
        pass

    def __init__(self, llm_modelid, rdf_input_file, quantize_bits=None):
        # init
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
        # RDF GRAPH INDED
        self.graph_indexer = RDFGraphIndexer("/mnt/newdrive/data_files/SpaTex/CSZt.nt")
        self.graph_indexer.build_index()
        # NER using spaCy
        self.entity_extractor = GeoEntityExtractor()

        # load model (TE and LLM)
        self.loadModel()

        # set system role
        self.generateSystemRole()