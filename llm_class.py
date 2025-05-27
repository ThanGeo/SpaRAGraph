import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from geo_extractor import GeoEntityExtractor
from rdflib import Graph, URIRef
from rdf_graph_class import RDFGraphIndexer
import re
import random

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    PURPLE = '\033[35m'
    MAGENTA = '\033[35m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

class REASONING:
    EXTERNAL = 0,
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
        # add the system role
        messages = self.getSystemRole() + messages
        print(messages)
        return self.generateAndDecode(messages)

    # def __init__(self, llm_modelid):
    #     self.llm_modelid = llm_modelid
    #     self.loadModel()
    #     self.generateSystemRole()

# Plain LLM
class PlainLLM(LLM):
    def generate(self, prompt, query_type):
        # messages = [{"role": "user", "content": "Question: " + prompt+"\n"}]
        # return super().generate(messages)
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
    
    
    ''' Works for STRUCTURED CONTEXT. If the context generation changes format, then this might fail'''
    def extract_triple(self, sentence):
        # Pattern 1: "X is Y of Z"
        match = re.match(r"(.+?) is (.+?) (?:of|with) (.+?)(?:$| and)", sentence)
        if match:
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(3).strip()
            if (obj[-1] == "."):
                obj = obj[:-1]
            return (subject, predicate, obj)
        
        # Pattern 2: "X contains Z..." (with optional trailing text)
        match = re.match(r"(.+?) contains (.+?)(?:$| and)", sentence)
        if match:
            subject = match.group(1).strip()
            predicate = "contains"
            obj = match.group(2).strip()
            return (subject, predicate, obj)

        # Pattern 3: "X intersects with Z..." (with optional trailing text)
        match = re.match(r"(.+?) intersects with (.+?)(?:$| and)", sentence)
        if match:
            subject = match.group(1).strip()
            predicate = "intersects with"
            obj = match.group(2).strip()
            return (subject, predicate, obj)

        
        print(f"No match for {sentence}")
        return None

    def _generate_yesno_examples(self, context, few_shot_num):
        few_shot_messages = []
        line = context.split('\n')[0]
        print(f"Context: {context}")
        
        if not line.strip():
            return []
        try:
            s, relation, o = self.extract_triple(line)
            print(f"Extracted: {s}, {relation}, {o}")
            if not relation:
                return [] 
        except:
            return []
                
        inverse_relation = self.graph_indexer.get_local_name(self.graph_indexer.getInverseRelation(relation))
        examples = []
        print(f"Relation: {relation}")
        print(f"inverse Relation: {inverse_relation}")
        
        for _ in range(few_shot_num):
            # Generate random unique entity letters for each example
            entity1, entity2 = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 2)
            
            # Create both the original and inverse relation examples
            examples.extend([
                {
                    "role": "user", 
                    "content": f"Context: Entity {entity1} is {relation} of Entity {entity2}.\nQuestion: Is {entity1} {inverse_relation} of {entity2}?\nAnswer with only 'yes' or 'no'."
                },
                {
                    "role": "assistant", 
                    "content": "no"
                },
                {
                    "role": "user", 
                    "content": f"Context: Entity {entity2} is {inverse_relation} of Entity {entity1}.\nQuestion: Is {entity2} {relation} of {entity1}?\nAnswer with only 'yes' or 'no'."
                },
                {
                    "role": "assistant", 
                    "content": "yes"
                }
            ])
        
        return examples[:few_shot_num*2]  # Return requested number (2 examples per iteration)

    def _get_options(self, prompt):
        pattern = r"Options:\s*(?:a\.\s*(.*?)\s*b\.\s*(.*?)\s*c\.\s*(.*?)\s*d\.\s*(.*?)\s*e\.\s*none of the above)"
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            a_option = match.group(1).strip()
            b_option = match.group(2).strip()
            c_option = match.group(3).strip()
            d_option = match.group(4).strip()
            
            if " of" in a_option:
                a_option.replace(" of", "")
            if " of" in b_option:
                b_option.replace(" of", "")
            if " of" in c_option:
                c_option.replace(" of", "")
            if " of" in d_option:
                d_option.replace(" of", "")
            # print(f"a. {a_option}")
            # print(f"b. {b_option}")
            # print(f"c. {c_option}")
            # print(f"d. {d_option}")
            # print("e. none of the above")
            
            return (a_option, b_option, c_option, d_option)

    def _generate_radio_examples(self, prompt, context, few_shot_num):
        few_shot_messages = []
        
        # Get options from the prompt
        options = self._get_options(prompt)
        if not options or len(options) != 4:
            return []
        
        a_option, b_option, c_option, d_option = options
        option_relations = [a_option, b_option, c_option, d_option]
        
        # Get all possible relations from the knowledge base
        all_relations = self.graph_indexer.getAllRelations()
        
        # Extract the single context line
        try:
            s, context_relation, o = self.extract_triple(context.strip())
            if not context_relation:
                return []
                        
            # Check if context relation matches any option
            relation_matches_option = any(context_relation in opt for opt in option_relations)
            
            # Generate multiple examples based on few_shot_num
            for _ in range(few_shot_num):
                # Generate random entity names for each example
                entity1, entity2 = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 2)
                
                # Format the relation text
                if "contains" not in context_relation:
                    formatted_relation = f"is {context_relation} of"
                else:
                    formatted_relation = context_relation
                
                # Case 1: Context relation matches one of the options (positive example)
                if relation_matches_option:
                    correct_option = None
                    for i, opt in enumerate(option_relations):
                        if context_relation in opt:
                            correct_option = chr(97 + i)
                            break
                    
                    example = [
                        {
                            "role": "user",
                            "content": f"Context: {s} {formatted_relation} {o}.\n"
                                    f"Question: Select exactly one option (a-e) that best describes the relationship of {s} in relation to {o}.\n"
                                    f"Options:\n"
                                    f"a. {a_option}\n"
                                    f"b. {b_option}\n"
                                    f"c. {c_option}\n"
                                    f"d. {d_option}\n"
                                    f"e. none of the above\n"
                                    "Instruction: Respond with only the single letter (a-e) corresponding to the correct option. "
                                    "Do not include any explanation or additional text."
                        },
                        {
                            "role": "assistant",
                            "content": correct_option
                        }
                    ]
                    few_shot_messages.extend(example)
                
                # Case 2: Context relation doesn't match any option (truth is 'e')
                else:
                    example = [
                        {
                            "role": "user",
                            "content": f"Context: {s} {formatted_relation} {o}.\n"
                                    f"Question: Select exactly one option (a-e) that best describes the relationship of {s} in relation to {o}.\n"
                                    f"Options:\n"
                                    f"a. {a_option}\n"
                                    f"b. {b_option}\n"
                                    f"c. {c_option}\n"
                                    f"d. {d_option}\n"
                                    f"e. none of the above\n"
                                    "Instruction: Respond with only the single letter (a-e) corresponding to the correct option. "
                                    "Do not include any explanation or additional text."
                        },
                        {
                            "role": "assistant",
                            "content": "e"
                        }
                    ]
                    few_shot_messages.extend(example)
                
                # Always add a negative example (with different relation not in options)
                invalid_relation = random.choice([
                    rel for rel in all_relations 
                    if not any(rel in opt for opt in option_relations) and rel != context_relation
                ])
                
                if "contains" not in invalid_relation:
                    invalid_relation = f"is {invalid_relation} of"
                
                negative_example = [
                    {
                        "role": "user",
                        "content": f"Context: {entity1} {invalid_relation} {entity2}.\n"
                                f"Question: Select exactly one option (a-e) that best describes the relationship of {entity1} in relation to {entity2}.\n"
                                f"Options:\n"
                                f"a. {a_option}\n"
                                f"b. {b_option}\n"
                                f"c. {c_option}\n"
                                f"d. {d_option}\n"
                                f"e. none of the above\n"
                                "Instruction: Respond with only the single letter (a-e) corresponding to the correct option. "
                                "Do not include any explanation or additional text."
                    },
                    {
                        "role": "assistant",
                        "content": "e"
                    }
                ]
                few_shot_messages.extend(negative_example)
                
        except Exception as e:
            print(f"Error generating examples: {e}")
            return []
        
        return few_shot_messages[:few_shot_num*4]  # 2 examples per iteration
        
    def _generate_checkbox_examples(self, prompt, context, few_shot_num):
        few_shot_messages = []
        
        # Get options from the prompt (these are entities in checkbox queries)
        options = self._get_options(prompt)
        if not options:
            return []
        
        try:
            # Split context into lines and process each line
            context_lines = [line.strip() for line in context.split('\n') if line.strip()]
            if not context_lines:
                return []
            
            # Get all possible relations from the knowledge base
            all_relations = self.graph_indexer.getAllRelations()
            
            # Extract relations from all context lines
            context_relations = set()
            subject = None
            for line in context_lines:
                try:
                    s, rel, o = self.extract_triple(line)
                    if rel:
                        context_relations.add(rel)
                        subject = s  # We'll use the last subject found
                except:
                    continue
            
            if not context_relations:
                return []
            
            # Generate multiple examples based on few_shot_num
            for _ in range(few_shot_num):
                # Generate random entity names for each example
                random_entity = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
                # For each example, we'll focus on one primary relation
                primary_relation = random.choice(list(context_relations))
                
                # Format the relation text
                if "contains" not in primary_relation:
                    formatted_relation = f"is {primary_relation} of"
                else:
                    formatted_relation = primary_relation
                
                # Build context string from all lines
                context_str = "\n".join([f"- {line}" for line in context_lines])
                
                # Determine correct options based on all context lines
                correct_options = []
                for i, option in enumerate(options):
                    # Check if any context line shows this relation with the option
                    # In real implementation, you would check your knowledge base
                    # Here we simulate with random chance, higher probability for options mentioned in context
                    option_in_context = any(option.lower() in line.lower() for line in context_lines)
                    if option_in_context and random.random() > 0.5 or random.random() > 0.8:
                        correct_options.append(chr(97 + i))
                
                # If no options are correct, we should still have at least one example
                if not correct_options:
                    correct_options = ['e']
                
                # Format correct options as comma-separated letters (e.g., "a, c")
                correct_answer = ", ".join(sorted(correct_options)) if 'e' not in correct_options else 'e'
                
                example = [
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\n"
                                f"Question: Based on the context, select all options (a-e) that {subject} is {primary_relation} of.\n"
                                f"Options:\n"
                                + "\n".join([f"{chr(97 + i)}. {opt}" for i, opt in enumerate(options)])
                                + f"\ne. none of the above\n"
                                "Instruction: Respond with only the letters (a-e) corresponding to all correct options, "
                                "separated by commas if multiple. Do not include any explanation or additional text."
                    },
                    {
                        "role": "assistant",
                        "content": correct_answer
                    }
                ]
                few_shot_messages.extend(example)
                
                # Case 2: Negative example with different relation not in context
                invalid_relation = random.choice([
                    rel for rel in all_relations 
                    if rel not in context_relations
                ])
                
                if "contains" not in invalid_relation:
                    invalid_relation = f"is {invalid_relation} of"
                
                negative_example = [
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\n"
                                f"Question: Based on the context, select all options (a-e) that {random_entity} is {invalid_relation}.\n"
                                f"Options:\n"
                                + "\n".join([f"{chr(97 + i)}. {opt}" for i, opt in enumerate(options)])
                                + f"\ne. none of the above\n"
                                "Instruction: Respond with only the letters (a-e) corresponding to all correct options, "
                                "separated by commas if multiple. Do not include any explanation or additional text."
                    },
                    {
                        "role": "assistant",
                        "content": "e"
                    }
                ]
                few_shot_messages.extend(negative_example)
                
        except Exception as e:
            print(f"Error generating checkbox examples: {e}")
            return []
        
        return few_shot_messages[:few_shot_num*4]  # 2 examples per iteration
    
    def getFewShotMessages(self, prompt, context, few_shot_num, qtype):
        if few_shot_num <= 0:
            return []
        
        # generative few-shot/context learning 
        # if qtype == "yes/no":
        #     return self._generate_yesno_examples(context, few_shot_num)
        # elif qtype == "radio":
        #     return self._generate_radio_examples(prompt, context, few_shot_num)
        # elif qtype == "checkbox":
        #     return self._generate_checkbox_examples(prompt, context, few_shot_num)

        # static few-shot learning
        examples = [
        {
            "role": "user", 
            "content": "Entity A is east of Entity B. Where is entity B relative to A?"
        },
        {
            "role": "assistant", 
            "content": "Entity B is west of Entity A."
        },
        {
            "role": "user", 
            "content": "Entity X is inside of Entity Y. Where is entity Y relative to X?"
        },
        {
            "role": "assistant", 
            "content": "Entity Y contains Entity X."
        },
        {
            "role": "user", 
            "content": "Entity <name1> is northwest of Entity <name2>. Where is entity <name2> relative to <name1>?"
        },
        {
            "role": "assistant", 
            "content": "Entity <name2> southeast Entity <name1>."
        },
        ]
        return examples[:2*few_shot_num]

    def _get_external_reasoning_context(self, pairs):
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
        
        return context
    
    def _chain_of_though_on_path(self, start, end, path):
        formatted_path = ""
        counter = 1
        for s, p, o in path:
            clean_s = self.graph_indexer.get_local_name(s)
            clean_p = self.graph_indexer.get_local_name(p)
            clean_o = self.graph_indexer.get_local_name(o)
            
            formatted_path += f"{counter}. {clean_s} is {clean_p} of {clean_o}.\n"
            
            counter += 1
        
        # get the reasoning    
        self.system_role = [{"role": "system", "content": "You are a spatial reasoning assistant. Always use the provided context to answer questions accurately."}]
        
        message = "Given the following spatial relations:\n" + formatted_path + f"What is the spatial relationship between {start} and {end}? Let's think step by step."
        messages = [{"role": "user", "content": f"{message}"}]
        reasoning_response = super().generate(messages)
        # summarize as a verdict (context)
        messages.append({"role": "assistant", "content": f"{reasoning_response}"})
        messages.append({"role": "user", "content": f"Summarize the spatial relationship between {start} and {end} in exactly one short sentence, without further explanation."})
        verdict = super().generate(messages)
        return verdict
    
    def _get_internal_reasoning_context(self, pairs):
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
                context = self._chain_of_though_on_path(clean_start, clean_end, path)
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
                    context = self._chain_of_though_on_path(clean_start, clean_end, path)
        
        return context
    
    def generate(self, prompt, query_type, few_shot_num, reasoning=REASONING.EXTERNAL):
        PROMPT = prompt
        message_text = ""
        # get entities
        entities = self.entity_extractor.extract_entities(prompt)
        # find start/end points in graph
        pairs = self.graph_indexer.find_start_end_pairs(entities)
        
        context = ""
        if reasoning == REASONING.EXTERNAL:
            # external reasoning
            context = self._get_external_reasoning_context(pairs)
        else:
            # inhouse reasoning
            context = self._get_internal_reasoning_context(pairs)
        
        messages = []
        
        if few_shot_num > 0:
            # add few shot examples
            messages += self.getFewShotMessages(prompt, context, few_shot_num, query_type)
        
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
        
            # for mistral, a single user message must be given.
            # message = f"Context: {context} {prompt}\nInstruction: Respond with only the single letter (a-e) corresponding to the correct option. Do not include any explanation or additional text."
            # messages.append({"role": "user", "content": f"{message}"})
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