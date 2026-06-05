from core.base_cg import BaseCG
from utils import CG_TYPE

from core.base_cm import BaseCM

import rdf_utils

class RDF_Composition(BaseCG):
    def __init__(self, cm: BaseCM) -> None:
        super().__init__(CG_TYPE.RDF_COMPOSITION)
        self.cm = cm

    def is_traversable(self, predicate: str) -> bool:
        """Delegate to the CM's traversal filter (if it has one)."""
        if hasattr(self.cm, "is_traversable"):
            return self.cm.is_traversable(predicate)
        return True

    def generateContext(self, paths: "list[list[(str,str,str)]]") -> str:
        context = ""
        for path in paths:
            # get the start and end entities for the entire path
            start = path[0][0]
            end = path[-1][2]
            # get combined relation
            combined_relation = self.cm.getCombinedRelation(path)
            # clean up start/end from URI to NL
            clean_start = rdf_utils.get_local_name(start)
            clean_end = rdf_utils.get_local_name(end)    
            # generate context
            context += rdf_utils.formatRDFtoNL(clean_start, combined_relation, clean_end) + "\n"

        return context