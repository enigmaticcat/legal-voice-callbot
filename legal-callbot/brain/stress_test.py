import asyncio
import json
import time
import argparse
from pathlib import Path
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime

# Mục tiêu: Đọc file JSONL chứa câu hỏi, gửi request mô phỏng nhiều người dùng ảo (concurrent users) cùng lúc tới server.
# Đo đạc các chỉ số: TTFT (Time To First Token), TPOT (Time Per Output Token), E2E (End-to-End Latency) và RAG search time.

async def send_request(session, url, query, session_id):
    """Gửi 1 request đến endpoint /think/stream và đo thời gian từng chunk."""
    payload = {
        "query": query,
        "session_id": session_id,
        "conversation_history": []
    }
    
    start_time = time.time()
    metrics = {
        "session_id": session_id,
        "query": query[:50] + "...", 
        "first_token_time": None,
        "end_to_end_time": None,
        "total_tokens": 0,
        "ttft": None, # Time To First Token (so với lúc bắt đầu gửi)
        "tpot": None, # Time Per Output Token (sau token đầu tiên)
        "rag_search_time": None,
        "success": False,
        "error": None
    }
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                metrics["error"] = f"HTTP {response.status}"
                return metrics
                
            first_chunk_received = False
            first_chunk_time = None
            last_chunk_time = None
            
            async for line in response.content:
                chunk_receive_time = time.time()
                last_chunk_time = chunk_receive_time
                metrics["total_tokens"] += 1
                
                if not first_chunk_received:
                    first_chunk_time = chunk_receive_time
                    metrics["first_token_time"] = first_chunk_time
                    metrics["ttft"] = first_chunk_time - start_time
                    first_chunk_received = True
                    
                # Parsing chunk data to extract timing if provided by server
                try:
                    line_data = json.loads(line)
                    if "timing" in line_data and "rag_search" in line_data["timing"]:
                        metrics["rag_search_time"] = line_data["timing"]["rag_search"]
                except:
                    pass
                    
            end_time = time.time()
            metrics["end_to_end_time"] = end_time - start_time
            metrics["success"] = True
            
            # Tính TPOT (Thời gian trung bình sinh 1 token sau token đầu tiên)
            # TPOT = (Thời gian nhận chunk cuối - Thời gian nhận chunk đầu) / (Số token - 1)
            if metrics["total_tokens"] > 1:
                tpot_duration = last_chunk_time - first_chunk_time
                metrics["tpot"] = tpot_duration / (metrics["total_tokens"] - 1)
                
    except Exception as e:
        metrics["error"] = str(e)
        
    return metrics

async def worker(worker_id, session, url, queries, reqs_per_worker, results):
    """Một worker mô phỏng 1 user gửi liên tiếp các request."""
    for i in range(reqs_per_worker):
        # Chọn ngẫu nhiên 1 câu hỏi từ tập test
        query = random.choice(queries) if queries else "Xin chào, vui lòng tư vấn giúp tôi."
        session_id = f"user_{worker_id}_req_{i}"
        
        print(f"[{time.strftime('%H:%M:%S')}] User {worker_id} đang gửi request {i+1}...")
        metrics = await send_request(session, url, query, session_id)
        results.append(metrics)
        
        # Nghỉ một chút giữa các request để giống người dùng thật (think time)
        await asyncio.sleep(random.uniform(0.5, 2.0))

async def main(args):
    url = f"http://{args.host}:{args.port}/think/stream"
    print(f"🚀 Bắt đầu Load Test trên endpoint: {url}")
    print(f"👥 Cấu hình: {args.users} concurrent users, {args.requests_per_user} requests/user")
    
    # 1. Đọc dữ liệu test (test_qa_dataset.jsonl)
    queries = []
    data_path = Path("data/test_qa_dataset.jsonl")
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    queries.append(data["instruction"])
                except Exception:
                    pass
        print(f"✅ Đã tải {len(queries)} câu hỏi từ {data_path}")
    else:
        print(f"⚠️ Không tìm thấy file {data_path}. Sử dụng các câu hỏi mặc định.")
        queries = [
            "Công ty tôi Cổ phần có 3 người, giờ muốn thêm 1 người nữa thì hồ sơ thay đổi ra sao?",
            "Vợ chồng em ly hôn mà có tranh chấp tài sản là căn nhà đứng tên chồng trước khi cưới thì chia sao luật sư?",
            "Tôi chạy xe máy bị cảnh sát giao thông phạt lỗi không gương chiếu hậu, mức phạt hiện hành là bao nhiêu?"
        ]

    # Warm-up (Khởi động hệ thống kết nối)
    print("🔥 Đang khởi động hệ thống (Warm-up 3 requests)...")
    async with aiohttp.ClientSession() as session:
        import random
        for _ in range(3):
            await send_request(session, url, random.choice(queries), "warmup")
    print("✅ Warm-up hoàn tất!\n")
    
    # 2. Chạy Load Test chính thức
    start_time = time.time()
    results = []
    
    timeout = aiohttp.ClientTimeout(total=60) # Timeout 60s cho mỗi request
    connector = aiohttp.TCPConnector(limit=args.users * 2) # Cho phép nhiều kết nối
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Tạo danh sách các worker chạy ĐỒNG THỜI (Concurrency)
        tasks = []
        for worker_id in range(args.users):
            task = asyncio.create_task(
                worker(worker_id, session, url, queries, args.requests_per_user, results)
            )
            tasks.append(task)
            
        # Chờ tất cả worker hoàn thành
        await asyncio.gather(*tasks)
        
    total_time = time.time() - start_time
    
    # 3. Tổng hợp và Vẽ Báo cáo Percentiles
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ LOAD TEST (LATENCY BENCHMARK)")
    print("="*50)
    print(f"Tổng thời gian chạy test: {total_time:.2f} giây")
    
    df = pd.DataFrame(results)
    success_df = df[df["success"] == True]
    failed_count = len(df) - len(success_df)
    print(f"Tổng số Requests: {len(df)} | Thành công: {len(success_df)} | Thất bại: {failed_count}")
    
    if len(success_df) == 0:
        print("❌ TOÀN BỘ REQUEST THẤT BẠI. Vui lòng kiểm tra lại server.")
        return
        
    # Tính toán các chỉ số thống kê (P50, P90, P99)
    metrics_to_report = ["ttft", "tpot", "end_to_end_time", "rag_search_time"]
    friendly_names = {
        "ttft": "Time To First Token (s)", 
        "tpot": "Time Per Output Token (s)", 
        "end_to_end_time": "End-to-End Latency (s)",
        "rag_search_time": "RAG Search Time (s)"
    }
    
    for metric in metrics_to_report:
        # Lọc ra các dòng có giá trị hợp lệ cho metric này
        valid_data = success_df[success_df[metric].notnull()][metric]
        if len(valid_data) == 0:
            continue
            
        print(f"\\n--- {friendly_names[metric]} ---")
        print(f"  • Trung bình (Mean): {valid_data.mean():.4f}s")
        print(f"  • Nhanh nhất (Min) : {valid_data.min():.4f}s")
        print(f"  • Chậm nhất (Max)  : {valid_data.max():.4f}s")
        print(f"  ► P50 (Median)     : {np.percentile(valid_data, 50):.4f}s  <-- Thường là trải nghiệm của 50% user")
        print(f"  ► P90              : {np.percentile(valid_data, 90):.4f}s  <-- 90% đánh giá là nhanh hơn mức này")
        print(f"  ► P99 (Tail)       : {np.percentile(valid_data, 99):.4f}s  <-- Kém nhất, có thể do cold-start")
        
    # Lưu file kết quả CSV để xem sau
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"latency_report_{timestamp}.csv"
    df.to_csv(report_file, index=False, encoding='utf-8')
    print(f"\\n📁 File chi tiết từng request đã được lưu vào: {report_file}")
    
    # 4. Đánh giá tự động cho mục tiêu Callbot Voice (< 1.5s E2E / < 1.0s TTFT)
    ttft_values = success_df["ttft"].dropna()
    ttft_p90 = np.percentile(ttft_values, 90) if not ttft_values.empty else 999
    
    print("\\n💡 ĐÁNH GIÁ CHUẨN VOICE AI:")
    if ttft_p90 < 1.0:
        print("  ✅ ĐẠT CHUẨN ĐỘ TRỄ: TTFT P90 < 1.0s. Độ phản hồi xuất sắc, phù hợp cho giọng nói thời gian thực.")
    elif ttft_p90 < 1.5:
        print("  ⚠️ CHẤP NHẬN ĐƯỢC: TTFT P90 < 1.5s. Hơi trễ một chút nhưng vẫn có thể dùng trong Voice Bot.")
    else:
        print(f"  ❌ KHÔNG NÊN DÙNG VOICE: TTFT P90 = {ttft_p90:.2f}s (vượt quá 1.5s). Cần tối ưu thêm Model hoặc RAG.")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load testing script for Legal CallBot Latency")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host của Brain Server")
    parser.add_argument("--port", type=int, default=50052, help="Port của Brain Server")
    parser.add_argument("-u", "--users", type=int, default=5, help="Số lượng người dùng gọi ĐỒNG THỜI (Concurrency)")
    parser.add_argument("-r", "--requests-per-user", type=int, default=10, help="Số lượng câu hỏi mỗi người dùng sẽ gửi")
    
    args = parser.parse_args()
    asyncio.run(main(args))
