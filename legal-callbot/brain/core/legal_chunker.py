import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("brain.core.legal_chunker")

# Hierarchical levels with their regex patterns and labels
# Higher index = Deeper level
HIERARCHY_LEVELS = [
    (re.compile(r'^(Phần|Chương)\s+([IVXLC\d]+)', re.I), "Phần/Chương"),
    (re.compile(r'^Mục\s+(\d+)', re.I), "Mục"),
    (re.compile(r'^Tiểu mục\s+(\d+)', re.I), "Tiểu mục"),
    (re.compile(r'^Điều\s+([a-zA-Z\d\.]+)', re.I), "Điều"),
    (re.compile(r'^([IVXLC]+)\.\s'), "Nhóm"),
    (re.compile(r'^(\d+)\.\s'), "Khoản"),
    (re.compile(r'^([a-zđ])\)\s'), "Điểm"),
    (re.compile(r'^([ivx]+)\)\s'), "Tiết"),
    (re.compile(r'^([-+*])\s'), "Gạch"),
]

def _word_count(text: str) -> int:
    return len(text.split())

class LegalChunker:
    def __init__(
        self,
        min_words: int = 80,
        max_words: int = 500,
        overlap_words: int = 30,
        use_contextual_enrichment: bool = True,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.use_contextual_enrichment = use_contextual_enrichment

    def extract_chunks(self, mapc: str, ten_entry: str, noidung: str, metadata: Dict) -> List[Dict]:
        """
        Extracts hierarchical chunks from a legal entry.
        Tracks state (breadcrumb) line-by-line.
        """
        noidung = (noidung or "").strip()
        
        # Parent record remains the same for reference
        parent_chunk = {
            "id": f"{mapc}_parent",
            "type": "parent",
            "mapc": mapc,
            "text": noidung,
            "ten_entry": ten_entry,
            "metadata": metadata,
        }

        # Step 1: Detect top-level breadcrumb from entry title (ten_entry)
        # e.g., "Điều 118..." -> [Điều 118]
        # e.g., "Mục 5..." -> [Mục 5]
        doc_name = metadata.get("demuc") or metadata.get("chude") or "Văn bản pháp luật"
        initial_breadcrumb = [doc_name]
        
        # Parse entry title for top-level hierarchy
        for regex, label in HIERARCHY_LEVELS[:4]: # Check Parte/Chapter/Section/Article
            m = regex.match(ten_entry)
            if m:
                initial_breadcrumb.append(f"{label} {m.group(2) if label == 'Phần/Chương' else m.group(1)}")
                break
        else:
            # If title doesn't match standard patterns, just use the title string
            initial_breadcrumb.append(ten_entry.split('.')[0])

        # Step 2: Process noidung line-by-line to extract segments with dynamic breadcrumbs
        segments = self._segment_by_hierarchy(noidung, initial_breadcrumb)
        
        # Step 3: Merge/Split segments based on size constraints while preserving breadcrumbs
        final_pieces = self._optimize_chunks(segments)
        
        # Step 4: Create final chunk objects with breadcrumb-enriched text
        children = []
        for idx, piece in enumerate(final_pieces):
            child = self._create_child_chunk(mapc, piece, idx, metadata)
            children.append(child)

        if not children:
            # Fallback if noidung is empty or too short
            children = [self._create_child_chunk(mapc, {"text": noidung, "breadcrumb": initial_breadcrumb}, 0, metadata)]

        return [parent_chunk] + children

    def _segment_by_hierarchy(self, text: str, initial_breadcrumb: List[str]) -> List[Dict]:
        """Splits text into segments, each tagged with its hierarchical breadcrumb."""
        lines = text.split('\n')
        segments = []
        
        # Current hierarchy stack
        # index in stack corresponds to index in HIERARCHY_LEVELS if match found
        # (level_index, label_text)
        stack = [] 

        for line in lines:
            line = line.strip()
            if not line: continue
            
            detected_level = -1
            label_val = ""
            
            for i, (regex, label) in enumerate(HIERARCHY_LEVELS):
                m = regex.match(line)
                if m:
                    detected_level = i
                    label_val = f"{label} {m.group(1) if i != 0 else m.group(1) + ' ' + m.group(2)}"
                    break
            
            if detected_level != -1:
                # Pop stack levels that are deeper or equal to detected
                while stack and stack[-1][0] >= detected_level:
                    stack.pop()
                stack.append((detected_level, label_val))
            
            # Combine initial breadcrumb with current stack
            current_breadcrumb = initial_breadcrumb + [s[1] for s in stack]
            
            # If same breadcrumb as last segment, just append text
            if segments and segments[-1]["breadcrumb"] == current_breadcrumb:
                segments[-1]["text"] += "\n" + line
            else:
                segments.append({
                    "text": line,
                    "breadcrumb": current_breadcrumb
                })
        
        return segments

    def _optimize_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Merges small segments and splits oversized ones while preserving breadcrumbs."""
        # 1. Merge small adjacent segments with SAME breadcrumb
        # (This is already mostly handled in _segment_by_hierarchy, but just in case)
        
        # 2. Merge small segments with different breadcrumbs if they are very short (e.g. unnumbered bullets)
        # We actually want to KEEP breadcrumbs as specific as possible, so we only merge 
        # if the total word count is still small.
        
        optimized = []
        if not segments: return []
        
        current = segments[0]
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            # If current is too small, try to merge forward
            if _word_count(current["text"]) < self.min_words:
                # Merge into next_seg but keep next_seg's breadcrumb as it's the "new" context
                # Actually, in Law, merging backward is safer for context.
                pass
            
            # Standard merge if same breadcrumb
            if current["breadcrumb"] == next_seg["breadcrumb"]:
                current["text"] += "\n\n" + next_seg["text"]
            else:
                # If current is very small (like a header), merge it into the next one
                if _word_count(current["text"]) < 20: 
                    next_seg["text"] = current["text"] + "\n" + next_seg["text"]
                    current = next_seg
                else:
                    optimized.append(current)
                    current = next_seg
        
        optimized.append(current)
        
        # 3. Handle oversized chunks (Split by sentence/words)
        final = []
        for piece in optimized:
            if _word_count(piece["text"]) <= self.max_words:
                final.append(piece)
            else:
                # Split large chunk but REPLICATE breadcrumb for all parts
                sub_texts = self._hard_split(piece["text"])
                for sub_t in sub_texts:
                    final.append({
                        "text": sub_t,
                        "breadcrumb": piece["breadcrumb"]
                    })
        
        return final

    def _hard_split(self, text: str) -> List[str]:
        """Split by sentence then words as last resort."""
        sentences = re.split(r'(?<=[.!?;])\s+', text)
        chunks = []
        buffer = ""
        
        for sentence in sentences:
            if _word_count(buffer + " " + sentence) <= self.max_words:
                buffer = (buffer + " " + sentence).strip()
            else:
                if buffer: chunks.append(buffer)
                if _word_count(sentence) > self.max_words:
                    # Word split
                    words = sentence.split()
                    for i in range(0, len(words), self.max_words):
                        chunks.append(" ".join(words[i:i+self.max_words]))
                    buffer = ""
                else:
                    buffer = sentence
        if buffer: chunks.append(buffer)
        return chunks

    def _create_child_chunk(self, mapc: str, piece: Dict, idx: int, metadata: Dict) -> Dict:
        breadcrumb = piece["breadcrumb"]
        text = piece["text"]
        
        # Breadcrumb examples:
        # [Mục 5] -> "[Mục 5]"
        # [Luật A, Điều 1, Khoản 2, Điểm a] -> "[Luật A - Điều 1 - Khoản 2 - Điểm a]"
        
        main_label = " - ".join(breadcrumb)
        
        # Aliasing for Khoản 2a style
        aliased_label = main_label
        if len(breadcrumb) >= 4:
            # Check if we have Khoản and Điểm
            if "Khoản" in breadcrumb[-2] and "Điểm" in breadcrumb[-1]:
                k_num = breadcrumb[-2].split()[-1]
                p_char = breadcrumb[-1].split()[-1].replace(')', '')
                aliased_label += f" (Khoản {k_num}{p_char})"

        enriched_text = f"[{aliased_label}] {text}"
        
        return {
            "id": f"{mapc}_child_{idx}",
            "type": "child",
            "parent_id": f"{mapc}_parent",
            "mapc": mapc,
            "text": enriched_text,
            "breadcrumb": breadcrumb,
            "metadata": metadata,
        }
