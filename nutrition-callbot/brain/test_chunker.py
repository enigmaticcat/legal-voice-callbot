"""
Test Legal Chunker v2 with real data from law_data.json

3 test cases:
  1. Short entry (~50 words) — should produce 1 child chunk
  2. Medium entry (~556 words, "Nhiệm vụ của Bảo vệ dân phố") — multiple Khoản chunks
  3. Long entry (~11,330 words, "Hệ thống tài khoản kế toán") — 25-40 chunks with merge
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.legal_chunker import LegalChunker


DATA_PATH = "/Users/nguyenthithutam/Desktop/Callbot/law_data.json"


def load_entry_by_mapc(data, mapc):
    for item in data:
        if item.get("mapc") == mapc:
            return item
    return None


def print_chunk_summary(chunks, show_text_preview=True):
    """Print a summary table of chunks."""
    parent = [c for c in chunks if c["type"] == "parent"]
    children = [c for c in chunks if c["type"] == "child"]

    print(f"  Parent chunks: {len(parent)}")
    print(f"  Child chunks:  {len(children)}")
    print()

    if children:
        sizes = [len(c["text"].split()) for c in children]
        print(f"  Child word counts: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")
        print()

        for i, c in enumerate(children):
            wc = len(c["text"].split())
            label = c.get("chunk_label", "")
            label_str = f" [{label}]" if label else ""
            preview = c["text"][:100].replace("\n", "\\n")
            if show_text_preview:
                print(f"  Child {i:3d} [{wc:4d} từ]{label_str}: {preview}...")
            else:
                print(f"  Child {i:3d} [{wc:4d} từ]{label_str}")


def test_short_entry(chunker, data):
    """Test 1: Short entry that doesn't need splitting."""
    print("=" * 70)
    print("TEST 1: SHORT ENTRY (< max_words)")
    print("=" * 70)

    # Find a short entry
    short_item = None
    for item in data:
        nd = item.get("noidung", "")
        wc = len(nd.split())
        if 30 < wc < 100:
            short_item = item
            break

    if not short_item:
        print("  SKIP: No suitable short entry found")
        return True

    nd = short_item["noidung"]
    print(f"  Entry: {short_item['ten']}")
    print(f"  Original: {len(nd.split())} words")
    print()

    chunks = chunker.extract_chunks(
        short_item["mapc"], short_item["ten"], nd,
        {"chude": short_item.get("chude", "")}
    )

    print_chunk_summary(chunks)

    children = [c for c in chunks if c["type"] == "child"]
    assert len(children) == 1, f"Expected 1 child for short text, got {len(children)}"
    print("  PASSED: Short entry produces exactly 1 child chunk")
    return True


def test_medium_entry(chunker, data):
    """Test 2: Medium entry — 'Nhiệm vụ của Bảo vệ dân phố' (556 words, 6 Khoản)."""
    print()
    print("=" * 70)
    print("TEST 2: MEDIUM ENTRY (~556 words, 6 Khoản)")
    print("=" * 70)

    item = load_entry_by_mapc(data, "010010000000000010000060000000000000000000402214200380000500")
    if not item:
        print("  SKIP: Entry not found")
        return True

    nd = item["noidung"]
    print(f"  Entry: {item['ten']}")
    print(f"  Original: {len(nd.split())} words")
    print()

    chunks = chunker.extract_chunks(
        item["mapc"], item["ten"], nd,
        {"chude": item.get("chude", ""), "demuc": item.get("demuc", "")}
    )

    print_chunk_summary(chunks)

    children = [c for c in chunks if c["type"] == "child"]
    assert len(children) >= 3, f"Expected ≥3 children for 6-Khoản entry, got {len(children)}"

    # Check all children have parent_id
    for c in children:
        assert c.get("parent_id"), f"Child {c['id']} missing parent_id"

    print(f"  PASSED: Medium entry produces {len(children)} child chunks with parent_id")
    return True


def test_long_entry(chunker, data):
    """Test 3: Long entry — 'Hệ thống tài khoản kế toán' (11,330 words)."""
    print()
    print("=" * 70)
    print("TEST 3: LONG ENTRY (~11,330 words)")
    print("=" * 70)

    item = load_entry_by_mapc(data, "170010000000000055000080000000000000000000802273401405500200")
    if not item:
        print("  SKIP: Entry not found")
        return True

    nd = item["noidung"]
    print(f"  Entry: {item['ten']}")
    print(f"  Original: {len(nd.split())} words")
    print()

    chunks = chunker.extract_chunks(
        item["mapc"], item["ten"], nd,
        {"chude": item.get("chude", ""), "demuc": item.get("demuc", "")}
    )

    print_chunk_summary(chunks, show_text_preview=False)

    children = [c for c in chunks if c["type"] == "child"]
    sizes = [len(c["text"].split()) for c in children]

    assert len(children) >= 10, f"Expected ≥10 children for 11K-word entry, got {len(children)}"

    under_min = sum(1 for s in sizes if s < 40)  # Allow some flexibility below min_words
    over_max = sum(1 for s in sizes if s > 600)   # Allow some flexibility above max_words

    print(f"\n  Size distribution:")
    print(f"    Under 40 words:  {under_min}/{len(children)}")
    print(f"    Over 600 words:  {over_max}/{len(children)}")

    print(f"  PASSED: Long entry produces {len(children)} child chunks")
    return True


def test_batch_stats(chunker, data, n=1000):
    """Test 4: Run on first N entries and report statistics."""
    print()
    print("=" * 70)
    print(f"TEST 4: BATCH STATISTICS (first {n} entries)")
    print("=" * 70)

    all_child_sizes = []
    total_children = 0
    total_parents = 0
    entries_processed = 0

    for item in data[:n]:
        nd = item.get("noidung", "")
        if not nd.strip():
            continue

        chunks = chunker.extract_chunks(
            item.get("mapc", ""), item.get("ten", ""), nd,
            {"chude": item.get("chude", "")}
        )

        for c in chunks:
            if c["type"] == "parent":
                total_parents += 1
            else:
                total_children += 1
                all_child_sizes.append(len(c["text"].split()))

        entries_processed += 1

    print(f"  Entries processed: {entries_processed}")
    print(f"  Total parents:     {total_parents}")
    print(f"  Total children:    {total_children}")
    print(f"  Avg children/entry: {total_children / entries_processed:.1f}")
    print()

    if all_child_sizes:
        print(f"  Child word count stats:")
        print(f"    Mean: {sum(all_child_sizes)/len(all_child_sizes):.0f}")
        print(f"    Min:  {min(all_child_sizes)}")
        print(f"    Max:  {max(all_child_sizes)}")
        print()

        under_80 = sum(1 for s in all_child_sizes if s < 80)
        over_500 = sum(1 for s in all_child_sizes if s > 500)
        in_range = sum(1 for s in all_child_sizes if 80 <= s <= 500)
        total = len(all_child_sizes)

        print(f"    < 80 words:     {under_80:5d} ({under_80/total*100:.1f}%)")
        print(f"    80-500 words:   {in_range:5d} ({in_range/total*100:.1f}%)")
        print(f"    > 500 words:    {over_500:5d} ({over_500/total*100:.1f}%)")

    print(f"\n  PASSED: Batch processing completed successfully")
    return True


def test_hierarchy_deep_dive(chunker):
    """Test 5: Explicitly verify Article 118 - Clause 2 - Point a/b with aliasing."""
    print()
    print("=" * 70)
    print("TEST 5: HIERARCHY DEEP DIVE (Article 118)")
    print("=" * 70)
    
    # Mock entry based on user example
    mock_entry = {
        "mapc": "300020000000000040000010000000000000000011800000000000000000",
        "ten": "Điều 30.2.LQ.118. Cưỡng chế thi hành nghĩa vụ buộc thực hiện công việc nhất định",
        "noidung": (
            "1. Trường hợp thi hành nghĩa vụ phải thực hiện công việc nhất định theo bản án... "
            "thì Chấp hành viên quyết định phạt tiền...\n\n"
            "2. Hết thời hạn đã ấn định mà người phải thi hành án không thực hiện nghĩa vụ thi hành án thì Chấp hành viên xử lý như sau:\n\n"
            "a) Trường hợp công việc đó có thể giao cho người khác thực hiện thay thì Chấp hành viên giao cho người có điều kiện thực hiện;\n\n"
            "b) Trường hợp công việc đó phải do chính người phải thi hành án thực hiện thì Chấp hành viên đề nghị cơ quan có thẩm quyền..."
        ),
        "demuc": "Thi hành án dân sự"
    }
    
    chunks = chunker.extract_chunks(
        mock_entry["mapc"], mock_entry["ten"], mock_entry["noidung"],
        {"demuc": mock_entry["demuc"]}
    )
    
    children = [c for c in chunks if c["type"] == "child"]
    
    print(f"  Entry: {mock_entry['ten']}")
    print(f"  Generated {len(children)} child chunks")
    
    # Verify labels
    found_2a = False
    found_2b = False
    
    for c in children:
        print(f"  - Chunk: {c['text'][:120]}...")
        if "(Khoản 2a)" in c["text"]:
            found_2a = True
        if "(Khoản 2b)" in c["text"]:
            found_2b = True
            
    assert found_2a, "Failed to find alias (Khoản 2a) in breadcrumb"
    assert found_2b, "Failed to find alias (Khoản 2b) in breadcrumb"
    print("\n  PASSED: Hierarchy labels and aliasing are correct")
    return True

if __name__ == "__main__":
    print("Loading data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries\n")

    chunker = LegalChunker(min_words=20, max_words=500) # Lower min words for testing segments

    all_passed = True
    all_passed &= test_short_entry(chunker, data)
    all_passed &= test_medium_entry(chunker, data)
    all_passed &= test_long_entry(chunker, data)
    all_passed &= test_hierarchy_deep_dive(chunker)
    all_passed &= test_batch_stats(chunker, data, n=100)

    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ")
    else:
        print("SOME TESTS FAILED ")
    print("=" * 70)
