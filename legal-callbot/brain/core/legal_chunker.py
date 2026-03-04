"""
Legal Document Chunker v2 — Smart Multi-Level Chunking for Vietnamese Legal Texts

Pipeline:  Smart Split → Merge Small → Hard Split Fallback → Add Overlap

Designed for law_data.json (74,490 entries) from Bộ Pháp Điển Điện Tử.
Compatible with BAAI/bge-m3 (8192 tokens) and bkai-foundation-models/vietnamese-bi-encoder (256 tokens).
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("brain.core.legal_chunker")

# ============================================================
# Split patterns ordered by Vietnamese legal hierarchy
# Each tuple: (compiled_regex, label_prefix)
# ============================================================
SPLIT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\n\n(?=\d+\.\s)'),           "Khoản"),        # 1. 2. 3.
    (re.compile(r'\n\n(?=[a-zđ]\)\s)'),          "Điểm"),         # a) b) c)
    (re.compile(r'\n\n(?=\d+\.\d+\.\s)'),       "Mục"),          # 3.1. 4.2.
    (re.compile(r'\n\n(?=\d+\.\d+\.\d+\.\s)'),  "Tiểu_mục"),     # 3.1.1.
    (re.compile(r'\n\n(?=[a-zđ]\.\s)'),          "Điểm_chấm"),    # a. b. c.
    (re.compile(r'\n\n(?=-\s)'),                 "Gạch"),         # - item
    (re.compile(r'\n\n(?=\+\s)'),                "Cộng"),         # + item
    (re.compile(r'\n\n'),                        "Đoạn"),         # paragraph break
]

# Labels for auto-detecting chunk label from content
LABEL_DETECTORS = [
    (re.compile(r'^(\d+)\.\s'),              lambda m: f"Khoản {m.group(1)}"),
    (re.compile(r'^([a-zđ])\)\s'),            lambda m: f"Điểm {m.group(1)}"),
    (re.compile(r'^(\d+\.\d+)\.\s'),         lambda m: f"Mục {m.group(1)}"),
    (re.compile(r'^(\d+\.\d+\.\d+)\.\s'),    lambda m: f"Tiểu mục {m.group(1)}"),
    (re.compile(r'^([a-zđ])\.\s'),            lambda m: f"Điểm {m.group(1)}"),
]


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _detect_chunk_label(text: str) -> str:
    """Auto-detect chunk label from the beginning of text content."""
    stripped = text.strip()
    for pattern, label_fn in LABEL_DETECTORS:
        m = pattern.match(stripped)
        if m:
            return label_fn(m)
    return ""


class LegalChunker:
    """
    Smart multi-level chunker for Vietnamese legal documents.
    
    Pipeline:
        1. Smart Split — tries legal structure patterns in priority order
        2. Merge Small — merges adjacent tiny chunks (< min_words) 
        3. Hard Split Fallback — splits oversized chunks by sentence/word
        4. Add Overlap — copies tail of chunk N to head of chunk N+1
    """
    
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

    # ==============================================================
    # Public API — same signature as v1 for backward compatibility
    # ==============================================================

    def extract_chunks(self, mapc: str, ten_dieu: str, noidung: str, metadata: Dict) -> List[Dict]:
        """
        Split one Điều into parent + child chunks.
        
        Returns:
            List[Dict] where [0] is the PARENT chunk (full text),
            and [1..N] are CHILD chunks (smaller pieces for embedding).
        """
        noidung = (noidung or "").strip()
        
        # Parent chunk: always the full original text
        parent_chunk = {
            "id": f"{mapc}_parent",
            "type": "parent",
            "mapc": mapc,
            "text": noidung,
            "ten_dieu": ten_dieu,
            "chunk_label": "",
            "metadata": metadata,
        }

        # If text is short enough, just create 1 child = parent
        if _word_count(noidung) <= self.max_words:
            child = self._make_child(mapc, ten_dieu, noidung, metadata, idx=0, label="")
            return [parent_chunk, child]

        # Run the 4-step pipeline
        raw_pieces = self._step1_smart_split(noidung)
        merged = self._step2_merge_small(raw_pieces)
        sized = self._step3_hard_split(merged)
        final_texts = self._step4_add_overlap(sized)

        # Build child chunk dicts
        children = []
        for idx, text in enumerate(final_texts):
            text = text.strip()
            if len(text) < 10:
                continue
            label = _detect_chunk_label(text)
            child = self._make_child(mapc, ten_dieu, text, metadata, idx=idx, label=label)
            children.append(child)

        # Fallback: if splitting produced nothing, use original text
        if not children:
            child = self._make_child(mapc, ten_dieu, noidung, metadata, idx=0, label="")
            children = [child]

        return [parent_chunk] + children

    # ==============================================================
    # Step 1: Smart Split — recursive pattern-based splitting
    # ==============================================================

    def _step1_smart_split(self, text: str) -> List[str]:
        """
        Try splitting with legal structure patterns in priority order.
        If a chunk is still > max_words after the current pattern,
        recursively try the next pattern on that chunk.
        """
        return self._recursive_split(text, pattern_idx=0)

    def _recursive_split(self, text: str, pattern_idx: int) -> List[str]:
        """Recursively split text using patterns starting at pattern_idx."""
        # Base case: text is small enough or no more patterns
        if _word_count(text) <= self.max_words:
            return [text]

        if pattern_idx >= len(SPLIT_PATTERNS):
            return [text]  # Will be handled by hard split in step 3

        regex, _label = SPLIT_PATTERNS[pattern_idx]
        parts = regex.split(text)

        # Filter empty parts
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            # This pattern didn't help, try next
            return self._recursive_split(text, pattern_idx + 1)

        # Recursively split any parts that are still too large
        result = []
        for part in parts:
            if _word_count(part) > self.max_words:
                result.extend(self._recursive_split(part, pattern_idx + 1))
            else:
                result.append(part)

        return result

    # ==============================================================
    # Step 2: Merge Small — combine adjacent tiny chunks
    # ==============================================================

    def _step2_merge_small(self, chunks: List[str]) -> List[str]:
        """Merge adjacent chunks that are < min_words, respecting max_words."""
        if not chunks:
            return chunks

        merged = []
        buffer = chunks[0]

        for i in range(1, len(chunks)):
            chunk = chunks[i]
            combined = buffer + "\n\n" + chunk
            combined_words = _word_count(combined)

            if _word_count(buffer) < self.min_words and combined_words <= self.max_words:
                # Buffer is too small, merge with next chunk
                buffer = combined
            elif _word_count(chunk) < self.min_words and combined_words <= self.max_words:
                # Next chunk is too small, absorb it into buffer
                buffer = combined
            else:
                # Buffer is big enough, save it and move on
                merged.append(buffer)
                buffer = chunk

        # Don't forget the last buffer
        merged.append(buffer)

        # Final pass: if last chunk is tiny, merge it with the previous one
        if len(merged) > 1 and _word_count(merged[-1]) < self.min_words:
            combined = merged[-2] + "\n\n" + merged[-1]
            if _word_count(combined) <= self.max_words * 1.2:  # Allow 20% overflow for last chunk
                merged[-2] = combined
                merged.pop()

        return merged

    # ==============================================================
    # Step 3: Hard Split Fallback — sentence/word-based splitting
    # ==============================================================

    def _step3_hard_split(self, chunks: List[str]) -> List[str]:
        """Split any remaining oversized chunks by sentences, then by words."""
        result = []
        for chunk in chunks:
            if _word_count(chunk) <= self.max_words:
                result.append(chunk)
            else:
                result.extend(self._split_by_sentences(chunk))
        return result

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split oversized text by sentence boundaries."""
        # Split by sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?;])\s+', text)

        chunks = []
        buffer = ""

        for sentence in sentences:
            candidate = (buffer + " " + sentence).strip() if buffer else sentence
            if _word_count(candidate) <= self.max_words:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                # If single sentence > max_words, split by words
                if _word_count(sentence) > self.max_words:
                    chunks.extend(self._split_by_words(sentence))
                    buffer = ""
                else:
                    buffer = sentence

        if buffer:
            chunks.append(buffer)

        return chunks

    def _split_by_words(self, text: str) -> List[str]:
        """Last resort: split by word count."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.max_words):
            chunk = " ".join(words[i:i + self.max_words])
            chunks.append(chunk)
        return chunks

    # ==============================================================
    # Step 4: Add Overlap — copy tail words between adjacent chunks
    # ==============================================================

    def _step4_add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap_words from end of chunk N to start of chunk N+1."""
        if self.overlap_words <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            overlap_size = min(self.overlap_words, len(prev_words))
            overlap_text = " ".join(prev_words[-overlap_size:])
            result.append(f"...{overlap_text}\n\n{chunks[i]}")

        return result

    # ==============================================================
    # Helpers
    # ==============================================================

    def _make_child(self, mapc: str, ten_dieu: str, text: str, metadata: Dict, idx: int, label: str) -> Dict:
        """Create a child chunk dict with contextual enrichment."""
        enriched_text = self._enrich_context(ten_dieu, text.strip())
        return {
            "id": f"{mapc}_child_{idx}",
            "type": "child",
            "parent_id": f"{mapc}_parent",
            "mapc": mapc,
            "text": enriched_text,
            "ten_dieu": ten_dieu,
            "chunk_label": label,
            "metadata": metadata,
        }

    def _enrich_context(self, ten_dieu: str, child_text: str) -> str:
        """Prefix child chunk with the Điều title for better embedding quality."""
        if not self.use_contextual_enrichment:
            return child_text

        # Extract meaningful name from structured title
        # e.g. "Điều 1.1.LQ.1. Phạm vi điều chỉnh" → "Phạm vi điều chỉnh"
        parts = ten_dieu.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1].strip()) > 0:
            short_name = parts[1].strip()
            return f"[{short_name}] {child_text}"

        return f"[{ten_dieu}] {child_text}"
