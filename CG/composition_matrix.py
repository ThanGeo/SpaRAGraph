import rdf_utils
from abc import ABC, abstractmethod

from core.base_cm import BaseCM
from utils import CM_TYPE

UNCERTAIN_TOKEN = "???"


# ─── Relation Family abstraction ─────────────────────────────────────────────

class RelationFamily(ABC):
    """
    A group of predicates that share a composition rule.

    To add support for a new relation type (e.g. isCapitalOf, isPartOf):
      1. Subclass RelationFamily and implement owns / normalize / compose.
      2. Set traversable = True (default) so the BFS follows edges of this type.
      3. Instantiate it and insert it into ApproximateCM.families
         *before* PassthroughFamily (which is always the last catch-all).

    The two optional overrides handle cross-family hops (e.g. directional
    followed by topological):
      - is_uncertain(a, b)              — flag known-ambiguous pairs for logging
      - compose_cross(a, b, next_family) — what happens when the next hop
                                           belongs to a different family
    """

    # Whether BFS should follow edges belonging to this family.
    # Set to False for the PassthroughFamily catch-all so that unknown predicates
    # (e.g. owl:sameAs, rdf:type) are not traversed.
    traversable: bool = True

    @abstractmethod
    def owns(self, predicate: str) -> bool:
        """Return True if this family is responsible for this (cleaned) predicate."""

    @abstractmethod
    def normalize(self, predicate: str) -> str:
        """Return the canonical form used for composition lookups."""

    @abstractmethod
    def compose(self, rel_a: str, rel_b: str) -> str:
        """Compose two within-family relations. May return UNCERTAIN_TOKEN."""

    def is_uncertain(self, rel_a: str, rel_b: str) -> bool:
        """Return True if this pair is known to be ambiguous (used for logging only)."""
        return False

    def compose_cross(self, rel_a: str, rel_b: str,
                      next_family: "RelationFamily") -> str:
        """
        Compose rel_a (from this family) with rel_b (from a different family).
        Default: keep the current relation — unknown hops are transparent.
        """
        return rel_a


# ─── Concrete families ───────────────────────────────────────────────────────

class DirectionalFamily(RelationFamily):
    """
    8-direction spatial composition: north, northeast, east, …
    Also handles SpaTex's 'adjacent to and <direction>' qualifier.
    """

    _PREFIX = "adjacent to and "

    _RELATIONS = frozenset({
        "north", "northeast", "east", "southeast",
        "south", "southwest", "west", "northwest",
    })

    # Pairs that are genuinely ambiguous — the table still returns an approximation.
    _UNCERTAIN = frozenset({
        ("north",     "south"),     ("south",     "north"),
        ("northeast", "southwest"), ("southwest", "northeast"),
        ("east",      "west"),      ("west",      "east"),
        ("southeast", "northwest"), ("northwest", "southeast"),
    })

    _TABLE = {
        ("north",     "north"):     "north",
        ("north",     "northeast"): "northeast",
        ("north",     "east"):      "northeast",
        ("north",     "southeast"): "east",
        ("north",     "south"):     "north",       # uncertain — approximate
        ("north",     "southwest"): "west",
        ("north",     "west"):      "northwest",
        ("north",     "northwest"): "northwest",

        ("northeast", "north"):     "northeast",
        ("northeast", "northeast"): "northeast",
        ("northeast", "east"):      "east",
        ("northeast", "southeast"): "east",
        ("northeast", "south"):     "east",
        ("northeast", "southwest"): "north",       # uncertain — approximate
        ("northeast", "west"):      "north",
        ("northeast", "northwest"): "north",

        ("east",      "north"):     "northeast",
        ("east",      "northeast"): "east",
        ("east",      "east"):      "east",
        ("east",      "southeast"): "southeast",
        ("east",      "south"):     "southeast",
        ("east",      "southwest"): "southeast",
        ("east",      "west"):      "north",       # uncertain — approximate
        ("east",      "northwest"): "north",

        ("southeast", "north"):     "east",
        ("southeast", "northeast"): "east",
        ("southeast", "east"):      "southeast",
        ("southeast", "southeast"): "southeast",
        ("southeast", "south"):     "south",
        ("southeast", "southwest"): "south",
        ("southeast", "west"):      "east",
        ("southeast", "northwest"): "east",        # uncertain — approximate

        ("south",     "north"):     "south",       # uncertain — approximate
        ("south",     "northeast"): "east",
        ("south",     "east"):      "southeast",
        ("south",     "southeast"): "south",
        ("south",     "south"):     "south",
        ("south",     "southwest"): "south",
        ("south",     "west"):      "southwest",
        ("south",     "northwest"): "west",

        ("southwest", "north"):     "west",
        ("southwest", "northeast"): "south",       # uncertain — approximate
        ("southwest", "east"):      "south",
        ("southwest", "southeast"): "south",
        ("southwest", "south"):     "southwest",
        ("southwest", "southwest"): "southwest",
        ("southwest", "west"):      "west",
        ("southwest", "northwest"): "west",

        ("west",      "north"):     "northwest",
        ("west",      "northeast"): "north",
        ("west",      "east"):      "north",       # uncertain — approximate
        ("west",      "southeast"): "east",
        ("west",      "south"):     "southwest",
        ("west",      "southwest"): "west",
        ("west",      "west"):      "west",
        ("west",      "northwest"): "northwest",

        ("northwest", "north"):     "northwest",
        ("northwest", "northeast"): "north",
        ("northwest", "east"):      "north",
        ("northwest", "southeast"): "east",        # uncertain — approximate
        ("northwest", "south"):     "west",
        ("northwest", "southwest"): "west",
        ("northwest", "west"):      "northwest",
        ("northwest", "northwest"): "northwest",
    }

    def owns(self, predicate: str) -> bool:
        return self.normalize(predicate) in self._RELATIONS

    def normalize(self, predicate: str) -> str:
        if predicate.startswith(self._PREFIX):
            return predicate[len(self._PREFIX):]
        return predicate

    def is_uncertain(self, rel_a: str, rel_b: str) -> bool:
        return (self.normalize(rel_a), self.normalize(rel_b)) in self._UNCERTAIN

    def compose(self, rel_a: str, rel_b: str) -> str:
        key = (self.normalize(rel_a), self.normalize(rel_b))
        return self._TABLE.get(key, UNCERTAIN_TOKEN)

    def compose_cross(self, rel_a: str, rel_b: str,
                      next_family: "RelationFamily") -> str:
        # Directional relations persist across topological or unknown hops.
        return rel_a


class TopologicalFamily(RelationFamily):
    """
    Topological relations: inside, contains, intersects with.
    Composing two topological steps is always uncertain.
    """

    _RELATIONS = frozenset({"inside", "contains", "intersects with"})

    def owns(self, predicate: str) -> bool:
        return predicate in self._RELATIONS

    def normalize(self, predicate: str) -> str:
        return predicate

    def is_uncertain(self, rel_a: str, rel_b: str) -> bool:
        return True

    def compose(self, rel_a: str, rel_b: str) -> str:
        return UNCERTAIN_TOKEN

    def compose_cross(self, rel_a: str, rel_b: str,
                      next_family: "RelationFamily") -> str:
        # After a topological step, whatever comes next takes over.
        return rel_b


class PassthroughFamily(RelationFamily):
    """
    Catch-all fallback for predicates not claimed by any other family.
    Preserves the predicate as-is with no composition logic.

    traversable = False means the BFS will not follow edges of unknown
    predicate types, preventing spurious multi-hop paths through unrelated
    relations (e.g. rdf:type, owl:sameAs, isCapitalOf before it has its
    own family defined).

    This family must always be last in ApproximateCM.families.
    """

    traversable: bool = False

    def owns(self, predicate: str) -> bool:
        return True  # matches everything

    def normalize(self, predicate: str) -> str:
        return predicate

    def compose(self, rel_a: str, rel_b: str) -> str:
        return rel_b  # last-seen relation wins

    def compose_cross(self, rel_a: str, rel_b: str,
                      next_family: "RelationFamily") -> str:
        return rel_b


# ─── Composition matrix dispatcher ───────────────────────────────────────────

class ApproximateCM(BaseCM):
    """
    Dispatches relation composition across a list of RelationFamily instances.

    Families are checked in order; the first whose owns() returns True handles
    the predicate. PassthroughFamily is always last as a catch-all.

    To add a new relation type, insert a RelationFamily instance into
    self.families before PassthroughFamily. No other code needs to change.
    """

    def __init__(self) -> None:
        super().__init__(CM_TYPE.APPROXIMATE)
        self.families = [
            DirectionalFamily(),
            TopologicalFamily(),
            PassthroughFamily(),  # must be last
        ]
        self.uncertain_log = []

    def _family_of(self, predicate: str) -> RelationFamily:
        for family in self.families:
            if family.owns(predicate):
                return family
        return self.families[-1]

    def is_traversable(self, predicate: str) -> bool:
        """Return True if the BFS should follow edges with this (cleaned) predicate.

        Delegates to the owning family's traversable flag. By default, all
        explicitly defined families are traversable; PassthroughFamily (catch-all
        for unknown predicates) is not, so rdf:type / owl:sameAs / future
        unregistered predicates are silently skipped during path search.
        """
        return self._family_of(predicate).traversable

    def getCombinedRelation(self, path) -> str:
        if not path:
            return ""

        if len(path) == 1:
            return rdf_utils.get_local_name(path[0][1])

        current = rdf_utils.get_local_name(path[0][1])
        current_family = self._family_of(current)

        for _, p, _ in path[1:]:
            next_rel = rdf_utils.get_local_name(p)
            next_family = self._family_of(next_rel)

            if current_family is next_family:
                if current_family.is_uncertain(current, next_rel):
                    self.uncertain_log.append((current, next_rel))
                current = current_family.compose(current, next_rel)
            else:
                current = current_family.compose_cross(current, next_rel, next_family)

            current_family = self._family_of(current)

        return current
