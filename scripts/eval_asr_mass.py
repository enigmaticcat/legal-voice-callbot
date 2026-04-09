import os
import sys
import glob
import json
import string
import jiwer
import csv
import time
import argparse
from typing import Dict, Tuple

# Enable relative import from nutrition-callbot module
callbot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nutrition-callbot')
asr_dir = os.path.join(callbot_dir, 'asr')
sys.path.append(callbot_dir)
sys.path.append(asr_dir)
from core.transcriber import Transcriber

import unicodedata
import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Chuẩn hoá Unicode (tránh lỗi font KHOẺ vs KHỎE)
    text = unicodedata.normalize('NFC', text)
    text = text.upper()
    
    # 2. Xóa các ký tự đặc biệt và dấu câu trước
    special_quotes = "“”‘’–"
    for q in special_quotes:
        text = text.replace(q, ' ')
        
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    text = ' '.join(text.split())
    
    # 3. Regex Mapping (Đồng bộ Ground Truth và ASR)
    replacements = {
        r'\b0\b': 'KHÔNG',
        r'\b1\b': 'MỘT',  
        r'\b2\b': 'HAI',
        r'\b3\b': 'BA',
        r'\b4\b': 'BỐN',
        r'\b5\b': 'NĂM',
        r'\b6\b': 'SÁU',
        r'\b7\b': 'BẢY',
        r'\b8\b': 'TÁM',
        r'\b9\b': 'CHÍN',
        r'\b10\b': 'MƯỜI',
        r'\b369\b': 'BA SÁU CHÍN',
        r'\bOMEGA3\b': 'OMEGA BA',
        r'\bOMEGA6\b': 'OMEGA SÁU',
        r'\bOMEGA9\b': 'OMEGA CHÍN',
        r'\bACID\b': 'AXIT',
        r'\bVITAMINA\b': 'VITAMIN A',
        r'\bCOVID19\b': 'CÔ VÍT MƯỜI CHÍN',
        r'\bCOVID\b': 'CÔ VÍT',
        r'\bOXI\b': 'Ô XI',
        r'\bOXY\b': 'Ô XI'
    }
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
        
    # 4. Chuẩn hoá lần cuối
    text = ' '.join(text.split())
    return text

def load_ground_truths(jsonl_dir: str) -> Dict[str, str]:
    gt_dict = {}
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "eval_split_*.jsonl"))
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    gt_id = data.get("id")
                    question = data.get("question")
                    if gt_id and question:
                        gt_dict[gt_id] = question
                except json.JSONDecodeError:
                    pass
    return gt_dict

def process_file_input(wav_file: str, transcriber: Transcriber, stream) -> str:
    import wave
    import numpy as np
    chunk_size = 1024
    sample_rate = 16000
    with wave.open(wav_file, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != sample_rate:
            return "" # Skip invalid formats
        
        chunk = wf.readframes(chunk_size)
        while chunk:
            transcriber.accept_wave_with_ttft(stream, chunk)
            chunk = wf.readframes(chunk_size)
            
    # FLUSH: Thêm khoảng lặng ảo 0.5s để ép model xả nốt các từ đang ngậm ở cuối câu
    tail = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while transcriber.recognizer.is_ready(stream):
        transcriber.recognizer.decode_stream(stream)
        
    res = transcriber.recognizer.get_result(stream)
    return res.text if hasattr(res, 'text') else str(res).strip()

def run_evaluation(wav_dir: str, eval_jsonl_dir: str, output_csv: str):
    print("Loading ground truths...")
    gt_dict = load_ground_truths(eval_jsonl_dir)
    print(f"Loaded {len(gt_dict)} ground truth sentences.")
    
    wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} WAV files to evaluate.")
    
    print("Initializing ASR Model...")
    transcriber = Transcriber()
    
    results = []
    
    print("Starting inference...")
    for idx, wav_file in enumerate(wav_files, 1):
        filename = os.path.basename(wav_file)
        gt_id = os.path.splitext(filename)[0]
        
        if gt_id not in gt_dict:
            # Skip if there's no ground truth mapping
            continue
            
        ground_truth_raw = gt_dict[gt_id]
        ground_truth_norm = normalize_text(ground_truth_raw)
        
        # Inference
        stream = transcriber.create_stream()
        start_time = time.time()
        hypothesis_raw = process_file_input(wav_file, transcriber, stream)
        inference_time = time.time() - start_time
        
        hypothesis_norm = normalize_text(hypothesis_raw)
        
        # Calc metrics
        try:
            wer = jiwer.wer(ground_truth_norm, hypothesis_norm)
        except Exception:
            wer = 1.0
            
        print(f"[{idx}/{len(wav_files)}] {gt_id} | WER: {wer:.2f}")
        
        results.append({
            "id": gt_id,
            "ground_truth_raw": ground_truth_raw,
            "ground_truth_norm": ground_truth_norm,
            "hypothesis_raw": hypothesis_raw,
            "hypothesis_norm": hypothesis_norm,
            "wer": round(wer, 4),
            "inference_time_s": round(inference_time, 3)
        })
    
    # Save to CSV
    if results:
        avg_wer = sum(r['wer'] for r in results) / len(results)
        print(f"\n--- EVALUATION COMPLETE ---")
        print(f"Total processed files: {len(results)}")
        print(f"Average WER: {avg_wer:.4f}")
        
        keys = results[0].keys()
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"Report saved to {output_csv}")
    else:
        print("No matches between WAV files and Ground Truth IDs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="../wav_16k", help="Directory containing WAV files")
    parser.add_argument("--jsonl_dir", default="../evaluation", help="Directory containing ground truth JSONL files")
    parser.add_argument("--output_csv", default="../asr_wer_report.csv", help="Output report file")
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wav_dir = os.path.abspath(os.path.join(base_dir, args.wav_dir))
    jsonl_dir = os.path.abspath(os.path.join(base_dir, args.jsonl_dir))
    out_csv = os.path.abspath(os.path.join(base_dir, args.output_csv))
    
    run_evaluation(wav_dir, jsonl_dir, out_csv)
