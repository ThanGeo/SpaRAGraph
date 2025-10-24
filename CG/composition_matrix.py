import rdf_utils

from core.base_cm import BaseCM
from utils import CM_TYPE

UNCERTAIN_TOKEN = "???"

class ApproximateCM(BaseCM):
    '''
    As per our TSAS submission.
    ''' 
    def __init__(self) -> None:
        super().__init__(CM_TYPE.APPROXIMATE)

        # ???
        self.uncertain_cases = {
            ('north', 'south'): UNCERTAIN_TOKEN,
            ('northeast', 'southwest'): UNCERTAIN_TOKEN,
            ('east', 'west'): UNCERTAIN_TOKEN,
            ('southeast', 'northwest'): UNCERTAIN_TOKEN, 
            ('south', 'north'): UNCERTAIN_TOKEN,    
            ('southwest', 'northeast'): UNCERTAIN_TOKEN,    
            ('west', 'east'): UNCERTAIN_TOKEN,  
            ('northwest', 'southeast'): UNCERTAIN_TOKEN, 
            
            ('inside', 'contains'): UNCERTAIN_TOKEN, 
            ('inside', 'intersects_with'): UNCERTAIN_TOKEN, 
            ('contains', 'inside'): UNCERTAIN_TOKEN, 
            ('inside', 'intersects_with'): UNCERTAIN_TOKEN, 
            ('intersects_with', 'inside'): UNCERTAIN_TOKEN, 
            ('intersects_with', 'contains'): UNCERTAIN_TOKEN, 
            ('intersects_with', 'intersects_with'): UNCERTAIN_TOKEN, 
        }

        self.composition_table_8dir = {
            ('north', 'north'): 'north',
            ('north', 'northeast'): 'northeast',
            ('north', 'east'): 'northeast',
            ('north', 'southeast'): 'east',
            ('north', 'south'): 'north',        # ???
            ('north', 'southwest'): 'west',
            ('north', 'west'): 'northwest',
            ('north', 'northwest'): 'northwest',

            ('northeast', 'north'): 'northeast',
            ('northeast', 'northeast'): 'northeast',
            ('northeast', 'east'): 'east',
            ('northeast', 'southeast'): 'east',
            ('northeast', 'south'): 'east',
            ('northeast', 'southwest'): 'north', # ???
            ('northeast', 'west'): 'north',
            ('northeast', 'northwest'): 'north',

            ('east', 'north'): 'northeast',
            ('east', 'northeast'): 'east',
            ('east', 'east'): 'east',
            ('east', 'southeast'): 'southeast',
            ('east', 'south'): 'southeast',
            ('east', 'southwest'): 'southeast',
            ('east', 'west'): 'north',  # ???
            ('east', 'northwest'): 'north',

            ('southeast', 'north'): 'east',
            ('southeast', 'northeast'): 'east',
            ('southeast', 'east'): 'southeast',
            ('southeast', 'southeast'): 'southeast',
            ('southeast', 'south'): 'south',
            ('southeast', 'southwest'): 'south',
            ('southeast', 'west'): 'east',
            ('southeast', 'northwest'): 'east', # ???

            ('south', 'north'): 'south',    # ???
            ('south', 'northeast'): 'east',
            ('south', 'east'): 'southeast',
            ('south', 'southeast'): 'south',
            ('south', 'south'): 'south',
            ('south', 'southwest'): 'south',
            ('south', 'west'): 'southwest',
            ('south', 'northwest'): 'west',

            ('southwest', 'north'): 'west',
            ('southwest', 'northeast'): 'south',    # ???
            ('southwest', 'east'): 'south',
            ('southwest', 'southeast'): 'south',
            ('southwest', 'south'): 'southwest',
            ('southwest', 'southwest'): 'southwest',
            ('southwest', 'west'): 'west',
            ('southwest', 'northwest'): 'west',

            ('west', 'north'): 'northwest',
            ('west', 'northeast'): 'north',
            ('west', 'east'): 'north',  # ???
            ('west', 'southeast'): 'east',
            ('west', 'south'): 'southwest',
            ('west', 'southwest'): 'west',
            ('west', 'west'): 'west',
            ('west', 'northwest'): 'northwest',

            ('northwest', 'north'): 'northwest',
            ('northwest', 'northeast'): 'north',
            ('northwest', 'east'): 'north',
            ('northwest', 'southeast'): 'east', # ???
            ('northwest', 'south'): 'west',
            ('northwest', 'southwest'): 'west',
            ('northwest', 'west'): 'northwest',
            ('northwest', 'northwest'): 'northwest',
        }

    def getCombinedRelation(self, path):
        if not path:
            return ""
        if len(path) > 1:
            # adjacency is usually obsolete after more than 1 hops
            clean_name = rdf_utils.get_local_name(path[0][1])
            if "adjacent to and " in clean_name:
                clean_name = clean_name.replace("adjacent to and ", "")
            current_relation = clean_name
            # print(f"{path[0][0]}, {path[0][1]}, {path[0][2]}")
            counter = 1
            
            for s, p, o in path[1:]:
                # print(f"Current: {current_relation}")
                clean_name = rdf_utils.get_local_name(p)
                if "adjacent to and " in clean_name:
                    clean_name = clean_name.replace("adjacent to and ", "")
                
                # print(f"{s}, {p}, {o}")
                # print(f"({current_relation}, {clean_name}) gives:")
                if (current_relation, clean_name) in self.uncertain_cases:
                    # self.uncertainCasesList.append((current_relation, clean_name))
                    self.uncertainCasesList.append((path[counter-1], path[counter]))
                    
                if current_relation == "inside" or current_relation == "contains" or current_relation == "intersects with":
                    # retain the same
                    current_relation = clean_name
                elif clean_name == "inside" or clean_name == "contains" or clean_name == "intersects with":
                    # retain the same
                    current_relation = current_relation
                else:
                    current_relation = self.composition_table_8dir[(current_relation, clean_name)]
                        
                # print(f"    {current_relation}")
                counter += 1
        else:
            # single hop jump, keep as it is
            current_relation = rdf_utils.get_local_name(path[0][1])
            
        return current_relation
    