from core.base_fewshot import BaseFewShot
from utils import FEW_SHOT_TYPE

from index.faiss_index import FAISSIndex

fewshot_examples = [

    # --- Cardinal Directions ---
    [
        {"role": "user", "content": "Context: Entity A is north of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is south of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is south of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is north of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is east of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is west of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is west of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is east of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is northeast of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is southwest of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is northwest of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is southeast of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is southeast of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is northwest of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is southwest of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is northeast of Entity A."}
    ],

    # --- Adjacency Relations ---
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and north of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is adjacent to and south of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and east of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is adjacent to and west of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and southwest of Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is adjacent to and northeast of Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A and Entity B share a border. Question: Are Entity A and Entity B adjacent?"},
        {"role": "assistant", "content": "Yes."}
    ],
    [
        {"role": "user", "content": "Context: Entity A and Entity B do not share a border. Question: Are they adjacent?"},
        {"role": "assistant", "content": "No."}
    ],

    # --- Topological Relations ---
    [
        {"role": "user", "content": "Context: Entity A is inside Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B contains Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A contains Entity B. Question: Where is Entity B relative to Entity A?"},
        {"role": "assistant", "content": "Entity B is inside Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A overlaps with Entity B. Question: What is the topological relation between A and B?"},
        {"role": "assistant", "content": "Entity A and Entity B partially overlap."}
    ],
    [
        {"role": "user", "content": "Context: Entity A and Entity B have no common area. Question: What is the topological relation between A and B?"},
        {"role": "assistant", "content": "Entity A and Entity B are disjoint."}
    ],
    [
        {"role": "user", "content": "Context: Entity A touches Entity B at the boundary. Question: What is their topological relation?"},
        {"role": "assistant", "content": "Entity A and Entity B are adjacent."}
    ],
    [
        {"role": "user", "content": "Context: Entity A completely surrounds Entity B. Question: What is the relation of A to B?"},
        {"role": "assistant", "content": "Entity A contains Entity B."}
    ],

    # --- Yes/No reasoning ---
    [
        {"role": "user", "content": "Context: Entity A is west of Entity B. Question: Is Entity B east of Entity A?"},
        {"role": "assistant", "content": "Yes."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is north of Entity B. Question: Is Entity A south of Entity B?"},
        {"role": "assistant", "content": "No."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is southeast of Entity B. Question: Is Entity B northwest of Entity A?"},
        {"role": "assistant", "content": "Yes."}
    ],
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and east of Entity B. Question: Is Entity B adjacent to and west of Entity A?"},
        {"role": "assistant", "content": "Yes."}
    ],

    # --- Radio-style reasoning (choose one) ---
    [
        {"role": "user", "content": "Context: Entity A is north of Entity B. Question (Radio): Select the correct relation of Entity B to Entity A. Options: (A) North (B) South (C) East (D) West"},
        {"role": "assistant", "content": "B) South"}
    ],
    [
        {"role": "user", "content": "Context: Entity A overlaps partially with Entity B. Question (Radio): What is their topological relation? Options: (A) Disjoint (B) Contains (C) Overlaps (D) Adjacent"},
        {"role": "assistant", "content": "C) Overlaps"}
    ],
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and west of Entity B. Question (Radio): Select the correct relation of Entity B to Entity A. Options: (A) West (B) East (C) South (D) North"},
        {"role": "assistant", "content": "B) East"}
    ],

    # --- Checkbox-style reasoning (multi-select) ---
    [
        {"role": "user", "content": "Context: Entity A is adjacent to and south of Entity B. Question (Checkbox): Select all that apply. Options: [ ] Entity A is north of Entity B [x] Entity B is north of Entity A [x] Entity A and B are adjacent [ ] Entity A contains Entity B"},
        {"role": "assistant", "content": "Entity B is north of Entity A; Entity A and B are adjacent."}
    ],
    [
        {"role": "user", "content": "Context: Entity A contains Entity B, and Entity C is adjacent to Entity A. Question (Checkbox): Which statements are true? Options: [x] Entity A contains Entity B [x] Entity C is adjacent to Entity A [ ] Entity B contains Entity A [ ] Entity A and C overlap"},
        {"role": "assistant", "content": "Entity A contains Entity B; Entity C is adjacent to Entity A."}
    ],
    [
        {"role": "user", "content": "Context: Entity A overlaps with Entity B and both are inside Region C. Question (Checkbox): Which of the following are true? Options: [x] Entity A and B overlap [x] Entity A is inside Region C [x] Entity B is inside Region C [ ] Entity A contains Region C"},
        {"role": "assistant", "content": "Entity A and B overlap; both are inside Region C."}
    ]
]



class ANNFewShot(BaseFewShot):
    def __init__(self, model_id="all-MiniLM-L6-v2"):
        super().__init__(FEW_SHOT_TYPE.ANN_SIMILAR)
        self.index = FAISSIndex(fewshot_examples, model_id)
    
    def getKshot(self, text: str, k: int):
        examples = self.index.retrieveK(text, k)
        messages = [msg for r in examples if "messages" in r for msg in r["messages"]]

        return messages

