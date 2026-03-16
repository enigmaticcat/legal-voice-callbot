import json
from collections import Counter

# Phân tích task1_output.json (QA)
with open('task1_output.json', 'r', encoding='utf-8') as f:
    task1_data = json.load(f)

difficulty_counts = Counter([item['difficulty'] for item in task1_data])
print('Task 1 (QA) - Difficulty distribution:')
for difficulty, count in difficulty_counts.items():
    print(f'  {difficulty}: {count} samples ({count/len(task1_data)*100:.1f}%)')

print()

# Phân tích task2_output.json (NLI)  
with open('task2_output.json', 'r', encoding='utf-8') as f:
    task2_data = json.load(f)

difficulty_counts = Counter([item['difficulty'] for item in task2_data])
print('Task 2 (NLI) - Difficulty distribution:')
for difficulty, count in difficulty_counts.items():
    print(f'  {difficulty}: {count} samples ({count/len(task2_data)*100:.1f}%)')

print()

# Phân tích task3_output.json (Legal Reasoning)
with open('task3_output.json', 'r', encoding='utf-8') as f:
    task3_data = json.load(f)

difficulty_counts = Counter([item['difficulty'] for item in task3_data])
print('Task 3 (Legal Reasoning) - Difficulty distribution:')
for difficulty, count in difficulty_counts.items():
    print(f'  {difficulty}: {count} samples ({count/len(task3_data)*100:.1f}%)')

print()
print('Total samples:', len(task1_data) + len(task2_data) + len(task3_data))

# Kiểm tra chất lượng một vài mẫu
print('\n=== QUALITY CHECK ===')

# Kiểm tra task1 - QA questions
sample = task1_data[0]
print(f'Task 1 sample structure complete: {all(k in sample for k in ["question_1", "question_2", "original_id", "source_mapc", "difficulty"])}')

# Kiểm tra task2 - NLI
sample = task2_data[0]  
print(f'Task 2 sample structure complete: {all(k in sample for k in ["nli_1", "nli_2", "original_id", "source_mapc", "difficulty"])}')

# Kiểm tra task3 - Legal Reasoning
sample = task3_data[0]
print(f'Task 3 sample structure complete: {all(k in sample for k in ["rules", "situation", "conclusions", "original_id", "source_mapc", "difficulty"])}')