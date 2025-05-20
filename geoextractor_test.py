
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

ner = None
matcher = None

def setupSpacy():
    global ner, matcher
    ner = spacy.load("en_core_web_sm")
    # Add entity labels (optional but helps visualization)
    ner.vocab.strings.add("ZIPCODE")
    ner.vocab.strings.add("STATE")
    ner.vocab.strings.add("COUNTY")
    # Initialize the Matcher
    matcher = Matcher(ner.vocab)
    
    ''' ZIPCODES '''
    # Regex pattern for 5-digit zipcodes
    zipcode_pattern = [{"TEXT": {"REGEX": r"^\d{5}$"}}]
    matcher.add("ZIPCODE", [zipcode_pattern])
    
    ''' STATES '''
    us_states = [
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
        'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
        'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
        'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
        'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode', 'tennessee',
        'texas', 'utah', 'vermont', 'virginia', 'washington', 'wisconsin',
        'wyoming',
        
    ]
    state_patterns = [[{"LOWER": {"IN": us_states}}], 
                    # Multi-word states
                    [{"LOWER": "new"}, {"LOWER": {"IN": ["york", "hampshire", "jersey", "mexico"]}}],
                    [{"LOWER": "north"}, {"LOWER": {"IN": ["carolina", "dakota"]}}],
                    [{"LOWER": "south"}, {"LOWER": {"IN": ["carolina", "dakota"]}}],
                    [{"LOWER": "west"}, {"LOWER": "virginia"}],
                    [{"LOWER": "rhode"}, {"LOWER": "island"}]]
    matcher.add("STATE", state_patterns)


    ''' COUNTIES '''
    # Match multi-word counties ending with County/Parish/Municipio/etc.
    county_suffixes = ["county", "parish", "municipio", "city", "borough", "census"]
    county_patterns = []
    for suffix in county_suffixes:
        # Match patterns like "Los Angeles County" or "St. John Parish"
        pattern = [
            {"POS": "PROPN", "OP": "+"},  # One or more proper nouns (multi-word)
            {"LOWER": suffix}
        ]
        county_patterns.append(pattern)

    matcher.add("COUNTY", county_patterns)

def setStartEnd(entities):
    pass

def extractEntities(prompt):
    global ner, matcher
    doc = ner(prompt)
    matches = matcher(doc)
    entities = {"zipcode": [], "state": [], "county": []}
    
    # First collect all matches and remove overlapping ones
    spans = []
    for match_id, start, end in matches:
        label = ner.vocab.strings[match_id]
        span = doc[start:end]
        spans.append((label, start, end, span))
    
    # Sort by start position (earliest first)
    spans.sort(key=lambda x: x[1])
    
    # Remove overlapping spans, keeping the longest matches
    filtered_spans = []
    i = 0
    while i < len(spans):
        current = spans[i]
        # Look ahead to find any overlapping spans
        j = i + 1
        while j < len(spans) and spans[j][1] < current[2]:
            # If the next span is longer, make it the current span
            if spans[j][2] > current[2]:
                current = spans[j]
            j += 1
        filtered_spans.append(current)
        i = j
    
    # Now process the filtered spans
    i = 0
    while i < len(filtered_spans):
        label, start, end, span = filtered_spans[i]
        
        if label == "ZIPCODE":
            entities["zipcode"].append(span.text)
            i += 1
            
        elif label == "STATE":
            # Handle multi-word states
            state_text = span.text
            j = i + 1
            # Check if next match is also STATE and adjacent
            while (j < len(filtered_spans) and 
                   filtered_spans[j][0] == "STATE" and
                   filtered_spans[j][1] == end):
                state_text += " " + filtered_spans[j][3].text
                end = filtered_spans[j][2]
                j += 1
            entities["state"].append(state_text)
            i = j
            
        elif label == "COUNTY":
            county_text = span.text
            # Look ahead for state within next 3 tokens
            j = i + 1
            state_text = ""
            while (j < len(filtered_spans) and 
                   filtered_spans[j][1] <= end + 3):
                if filtered_spans[j][0] == "STATE":
                    state_text = filtered_spans[j][3].text
                    # Check for multi-word state
                    k = j + 1
                    while (k < len(filtered_spans) and 
                           filtered_spans[k][0] == "STATE" and
                           filtered_spans[k][1] == filtered_spans[j][2]):
                        state_text += " " + filtered_spans[k][3].text
                        k += 1
                    break
                j += 1
            
            if state_text:
                entities["county"].append(f"{county_text} {state_text}")
                # Skip the state we just processed
                while (i < len(filtered_spans) and 
                       filtered_spans[i][1] < filtered_spans[j][2]):
                    i += 1
            else:
                entities["county"].append(county_text)
                i += 1
    
    print(entities)
    
    
    

# setup
setupSpacy()

prompts = [
            # "The State of West Virginia contains Zipcode 24888, correct?",
            # "Is Zipcode 62546 northeast of Zipcode 62560?",
            # "Query: Select exactly one option (a-e) that best describes the relationship of Sanborn County South Dakota in relation to Zipcode 57363 in terms of geography. Options: a. intersects with b. they intersect c. adjacent to and northeast of d. adjacent to and north of e. none of the above",
            "Query: Select all options that intersect with Albemarle County Virginia? You may choose one or more options. Options: a. Zipcode 22902 b. Zipcode 24590 c. Zipcode 22920 d. Zipcode 22968 e. None of the above",
            # "Query: Select exactly one option (a-e) that best describes the relationship of Saint Martin Parish Louisiana in relation to Pointe Coupee Parish Louisiana in terms of geography. Options: a. adjacent to and southwest of b. northeast of c. adjacent to and south of d. north of e. none of the above",
           ]

for prompt in prompts:
    extractEntities(prompt)