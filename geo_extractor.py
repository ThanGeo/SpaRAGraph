import spacy
from spacy.matcher import Matcher

class GeoEntityExtractor:
    def __init__(self, bidirectional=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.bidirectional = bidirectional
        self._setup_patterns()

        
    def _setup_patterns(self):
        """Initialize all matching patterns"""
        # Add entity labels
        self.nlp.vocab.strings.add("ZIPCODE")
        self.nlp.vocab.strings.add("STATE")
        self.nlp.vocab.strings.add("COUNTY")
        
        # ZIPCODE pattern
        zipcode_pattern = [{"TEXT": {"REGEX": r"^\d{5}$"}}]
        self.matcher.add("ZIPCODE", [zipcode_pattern])
        
        # STATE patterns
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
        state_patterns = [
            [{"LOWER": {"IN": us_states}}],
            [{"LOWER": "new"}, {"LOWER": {"IN": ["york", "hampshire", "jersey", "mexico"]}}],
            [{"LOWER": "north"}, {"LOWER": {"IN": ["carolina", "dakota"]}}],
            [{"LOWER": "south"}, {"LOWER": {"IN": ["carolina", "dakota"]}}],
            [{"LOWER": "west"}, {"LOWER": "virginia"}],
            [{"LOWER": "rhode"}, {"LOWER": "island"}]
        ]
        self.matcher.add("STATE", state_patterns)
        
        # COUNTY patterns
        county_suffixes = ["county", "parish", "municipio", "city", "borough", "census"]
        county_patterns = []
        for suffix in county_suffixes:
            pattern = [
                {"POS": "PROPN", "OP": "+"},
                {"LOWER": suffix}
            ]
            county_patterns.append(pattern)
        self.matcher.add("COUNTY", county_patterns)
    
    
    def extract_entities(self, text):
        """Extract geographic entities from text in the order they appear"""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        # Create a list of all matches with their start position and type
        all_matches = []
        for match_id, start, end in matches:
            label = doc.vocab.strings[match_id]
            span = doc[start:end]
            all_matches.append({
                'label': label,
                'start': start,
                'end': end,
                'text': span.text,
                'span': span
            })
        
        # Sort all matches by their start position
        all_matches.sort(key=lambda x: x['start'])
        
        # Process multi-word entities and combine them
        processed_matches = []
        i = 0
        while i < len(all_matches):
            current = all_matches[i]
            
            if current['label'] == 'STATE':
                # Check for multi-word states
                j = i + 1
                while (j < len(all_matches) and 
                    all_matches[j]['label'] == 'STATE' and 
                    all_matches[j]['start'] == current['end']):
                    current['text'] += ' ' + all_matches[j]['text']
                    current['end'] = all_matches[j]['end']
                    j += 1
                
                # Only add as a state if it's not part of a county name
                is_part_of_county = False
                # Look ahead to see if this state is part of a county reference
                if j < len(all_matches) and all_matches[j]['label'] == 'COUNTY':
                    if all_matches[j]['start'] <= current['end'] + 3:  # within a small window
                        is_part_of_county = True
                
                if not is_part_of_county:
                    current['text'] = "The State of " + current['text']
                    processed_matches.append(current)
                i = j
                
            elif current['label'] == 'COUNTY':
                # Check for following state within a small window
                county_text = current['text']
                state_text = ''
                j = i + 1
                
                # Look forward for a state (within 3 tokens)
                while (j < len(all_matches)) and (all_matches[j]['start'] <= current['end'] + 3):
                    if all_matches[j]['label'] == 'STATE':
                        state_match = all_matches[j]
                        # Handle multi-word states
                        k = j + 1
                        while (k < len(all_matches)) and (all_matches[k]['label'] == 'STATE') and (all_matches[k]['start'] == state_match['end']):
                            state_match['text'] += ' ' + all_matches[k]['text']
                            state_match['end'] = all_matches[k]['end']
                            k += 1
                        state_text = state_match['text']
                        j = k  # skip the processed state matches
                        break
                    j += 1
                
                # If bidirectional is enabled, also look backward for states
                if self.bidirectional and not state_text:
                    k = i - 1
                    while (k >= 0) and (all_matches[k]['end'] >= current['start'] - 3):
                        if all_matches[k]['label'] == 'STATE':
                            state_match = all_matches[k]
                            # Handle multi-word states going backward
                            l = k - 1
                            while (l >= 0) and (all_matches[l]['label'] == 'STATE') and (all_matches[l]['end'] == state_match['start']):
                                state_match['text'] = all_matches[l]['text'] + ' ' + state_match['text']
                                state_match['start'] = all_matches[l]['start']
                                l -= 1
                            state_text = state_match['text']
                            k = l  # skip the processed state matches
                            break
                        k -= 1
                
                if state_text:
                    processed_matches.append({
                        'label': 'COUNTY',
                        'start': current['start'],
                        'end': current['end'],
                        'text': f"{county_text} {state_text}"
                    })
                    i = j if state_text else i + 1
                else:
                    processed_matches.append(current)
                    i += 1
                    
            else:  # ZIPCODE or others
                processed_matches.append(current)
                i += 1
        
        # Now create both the ordered entities list and categorized dictionary
        entities_dict = {"zipcode": [], "state": [], "county": []}
        ordered_entities = []
        
        for match in processed_matches:
            entity_info = {
                'text': match['text'],
                'type': match['label'].lower(),
                'start': match['start'],
                'end': match['end']
            }
            ordered_entities.append(entity_info)
            
            if match['label'] == 'ZIPCODE':
                entities_dict['zipcode'].append(match['text'])
            elif match['label'] == 'STATE':
                entities_dict['state'].append(match['text'])
            elif match['label'] == 'COUNTY':
                entities_dict['county'].append(match['text'])
        
        return {
            'categorized': entities_dict,
            'ordered': ordered_entities
        }
    
    def _collect_and_filter_matches(self, doc, matches):
        """Collect and filter overlapping matches"""
        spans = []
        for match_id, start, end in matches:
            label = doc.vocab.strings[match_id]
            span = doc[start:end]
            spans.append((label, start, end, span))
        
        spans.sort(key=lambda x: x[1])
        
        filtered_spans = []
        i = 0
        while i < len(spans):
            current = spans[i]
            j = i + 1
            while j < len(spans) and spans[j][1] < current[2]:
                if spans[j][2] > current[2]:
                    current = spans[j]
                j += 1
            filtered_spans.append(current)
            i = j
            
        return filtered_spans
    
    def _process_spans(self, doc, spans, entities):
        """Process filtered spans to extract entities"""
        i = 0
        while i < len(spans):
            label, start, end, span = spans[i]
            
            if label == "ZIPCODE":
                entities["zipcode"].append(span.text)
                i += 1
                
            elif label == "STATE":
                state_text, i = self._process_multi_word_state(spans, i)
                entities["state"].append(state_text)
                
            elif label == "COUNTY":
                county_text, state_text, i = self._process_county_with_state(doc, spans, i)
                if state_text:
                    entities["county"].append(f"{county_text} {state_text}")
                else:
                    entities["county"].append(county_text)
    
    def _process_multi_word_state(self, spans, current_idx):
        """Handle multi-word states"""
        state_text = spans[current_idx][3].text
        j = current_idx + 1
        end = spans[current_idx][2]
        
        while (j < len(spans) and (spans[j][0] == "STATE") and (spans[j][1] == end)):
            state_text += " " + spans[j][3].text
            end = spans[j][2]
            j += 1
            
        return state_text, j
    
    def _process_county_with_state(self, doc, spans, current_idx):
        """Handle county with potential following state"""
        county_text = spans[current_idx][3].text
        state_text = ""
        j = current_idx + 1
        
        while (j < len(spans)) and (spans[j][1] <= spans[current_idx][2] + 3):
            if spans[j][0] == "STATE":
                state_text = spans[j][3].text
                k = j + 1
                while (k < len(spans)) and (spans[k][0] == "STATE") and (spans[k][1] == spans[j][2]):
                    state_text += " " + spans[k][3].text
                    k += 1
                break
            j += 1
        
        if state_text:
            while (current_idx < len(spans)) and (spans[current_idx][1] < spans[j][2]):
                current_idx += 1
        else:
            current_idx += 1
            
        return county_text, state_text, current_idx

