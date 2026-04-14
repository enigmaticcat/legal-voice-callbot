# -*- coding: utf-8 -*-
import json, hashlib, uuid
from pathlib import Path

OUTPUT_FILE = Path('/Users/nguyenthithutam/Desktop/Callbot/data_final/corpus_final.jsonl')

def make_id(s):
    h = hashlib.md5(s.encode()).hexdigest()
    return str(uuid.UUID(h))

entries = [
    {
        'qid': 'thucuc_s5_087',
        'question': 'Sau mổ trĩ nên ăn gì và kiêng gì để nhanh hồi phục?',
        'url': 'https://benhvienthucuc.vn/song-khoe/benh-hau-mon-truc-trang/sau-khi-phau-thuat-tri-nen-an-gi-va-kieng-gi',
        'text': (
            'Chế độ ăn sau phẫu thuật trĩ theo từng giai đoạn: '
            'Ngày đầu tiên (24 giờ đầu): không nên ăn gì, cơ thể cần thời gian phục hồi từ gây mê và phẫu thuật. '
            'Tuần đầu (ngày 1–7): bắt đầu ăn thức ăn mềm, dễ tiêu — cháo, súp, trứng luộc, cá hấp, rau nấu mềm, sữa chua. '
            'Tránh thực phẩm cay nóng, nhiều dầu mỡ, chất xơ thô. '
            'Tuần thứ hai (ngày 7–14): mở rộng dần chế độ ăn — tăng chất xơ từ từ qua rau và trái cây, '
            'bổ sung protein nhẹ như thịt gà, cá; tiếp tục tránh thực phẩm kích thích đường tiêu hóa, '
            'uống đủ 2–2,5 lít nước/ngày để giữ phân mềm. '
            'Thực phẩm cần kiêng hoàn toàn: rượu bia (đặc biệt 24h đầu sau gây mê), đồ cay (ớt, tiêu, mù tạt), '
            'ngũ cốc tinh chế, thức ăn nhiều muối và đường, sản phẩm từ sữa (trừ sữa chua). '
            'Nên ăn thực phẩm giàu omega-3 (cá, hạt lanh, hạt chia) để chống viêm; '
            'bổ sung vitamin C và E (ổi, kiwi, rau cải) để hỗ trợ lành vết thương.'
        ),
    },
    {
        'qid': 'vdd_s4_080',
        'question': 'Để xương chắc khỏe, canxi cần được đưa vào khẩu phần như thế nào?',
        'url': 'https://www.vinmec.com/vie/bai-viet/canxi-giup-xay-he-xuong-chac-khoe-nhu-the-nao-vi',
        'text': (
            'Canxi và việc xây dựng khung xương chắc khỏe: '
            '99% lượng canxi trong cơ thể tập trung ở xương và răng. '
            'Khung xương phát triển nhanh nhất trong giai đoạn 10–20 tuổi (tuổi dậy thì và vị thành niên); '
            'mật độ xương đạt đỉnh (peak bone mass — vốn xương) vào khoảng 25–30 tuổi. '
            'Vốn xương tích lũy trong giai đoạn dậy thì đến 20 tuổi là nền tảng sức khỏe xương suốt cả đời — '
            'thiếu canxi trong thời kỳ này sẽ làm giảm mật độ xương đỉnh, tăng nguy cơ loãng xương khi về già. '
            'Nhu cầu canxi theo độ tuổi: '
            '1–3 tuổi: 500mg/ngày; 4–8 tuổi: 800mg/ngày; 9–18 tuổi: 1.300mg/ngày (cao nhất vì đang xây khung xương); '
            '19–50 tuổi: 1.000mg/ngày; từ 51 tuổi trở lên: 1.200mg/ngày. '
            'Cách đưa canxi vào khẩu phần: ưu tiên thực phẩm tự nhiên — sữa, sữa chua, phô mai, '
            'cá nhỏ ăn cả xương, rau xanh đậm (cải xoăn, bông cải xanh), đậu phụ; '
            'kết hợp vitamin D (phơi nắng 10–15 phút, 3 lần/tuần hoặc bổ sung 400–600 IU/ngày với người lớn tuổi) '
            'để hấp thu canxi vào xương hiệu quả. '
            'Tập thể dục chịu lực (đi bộ, chạy, nâng tạ) giúp duy trì mật độ xương và hạn chế mất canxi.'
        ),
    },
]

with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
    for e in entries:
        qid = e['qid']
        url = e['url']
        text = e['text']
        row = {
            'chunk_id': make_id(f'verified2_{qid}_{url}'),
            'doc_id': make_id(url),
            'source': 'manual_verified',
            'url': url,
            'title': '',
            'category': 'manual_verified',
            'chunk_index': 1,
            'text': text,
            'embed_text': text,
            'from_question': e['question'],
            'question_id': qid,
            'relevance_score': 0.95,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Appended: {qid} | {len(text)} chars')

print('Done')
