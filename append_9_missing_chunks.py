# -*- coding: utf-8 -*-
"""
Append 9 manual_verified chunks cho các câu MISSING sau leakage cleanup.
Nguồn: Vinmec, HelloBacsi, SucKhoeDoisong (đã fetch và tổng hợp).
"""
import json, hashlib, uuid
from pathlib import Path

OUTPUT_FILE = Path('/Users/nguyenthithutam/Desktop/Callbot/data_final/corpus_final.jsonl')

def make_id(s):
    h = hashlib.md5(s.encode()).hexdigest()
    return str(uuid.UUID(h))

entries = [
    # ------------------------------------------------------------------
    # thucuc_s1_066: mướp đắng khô nấu nước uống, tiểu đường type 2
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s1_066',
        'question': 'Mướp đắng khô nấu nước uống hỗ trợ tiểu đường type 2 có an toàn không?',
        'url': 'https://www.vinmec.com/vie/bai-viet/muop-dang-va-benh-tieu-duong-vi',
        'text': (
            'Mướp đắng (khổ qua, Momordica charantia) và bệnh tiểu đường type 2: '
            'Mướp đắng chứa lectin, charantin, polypeptid-P và vicine — các hoạt chất có tác dụng '
            'tương tự insulin, giúp tế bào hấp thu glucose hiệu quả hơn, chuyển vận glucose đến cơ và gan, '
            'và giảm chuyển hóa chất dinh dưỡng thành glucose. '
            'Cách dùng: mướp đắng khô (đã sao) pha trà uống hàng ngày — cho 5–7 lát vào 350 ml nước nóng, '
            'hãm 5–10 phút; hoặc sắc nước uống. Liều khuyến cáo: không quá 2,5 lạng (khoảng 2 quả) mỗi ngày. '
            'Lưu ý: mướp đắng KHÔNG được FDA hoặc Bộ Y tế phê duyệt là thuốc điều trị tiểu đường; '
            'chỉ dùng hỗ trợ, không thay thế thuốc kê đơn. '
            'Khi kết hợp với insulin hoặc thuốc hạ đường huyết, nguy cơ tụt đường huyết quá mức. '
            'Chống chỉ định: phụ nữ mang thai (có thể kích thích co bóp tử cung, gây sảy thai), '
            'người thiếu men G6PD (hồng cầu dễ vỡ), người sắp phẫu thuật. '
            'Tác dụng phụ có thể gặp: tiêu chảy, nôn mửa, đau bụng nếu dùng quá liều. '
            'Kết luận: người tiểu đường type 2 có thể dùng mướp đắng khô như thực phẩm hỗ trợ, '
            'nhưng phải theo dõi đường huyết thường xuyên và báo bác sĩ điều trị.'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s2_076: sau sinh mổ hơn tuần ăn trứng gà được không
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s2_076',
        'question': 'Sau sinh mổ hơn tuần có ăn trứng gà được không, sẹo lồi ra sao?',
        'url': 'https://www.vinmec.com/vie/bai-viet/che-do-uong-va-dinh-duong-hop-ly-sau-sinh-mo-vi',
        'text': (
            'Trứng gà sau sinh mổ — thực hư sẹo lồi và hướng dẫn dinh dưỡng: '
            'Sau sinh mổ, sản phụ hoàn toàn có thể ăn trứng gà vì trứng là nguồn protein chất lượng cao '
            'cung cấp axit amin thiết yếu để tái tạo mô và lành vết thương, cùng vitamin A, D, B12, sắt, kẽm, choline '
            'hỗ trợ miễn dịch và tạo sữa. '
            'Về sẹo lồi: chưa có nghiên cứu y khoa nào chứng minh trứng gà trực tiếp gây sẹo lồi. '
            'Tuy nhiên, nên kiêng ăn lòng trắng trứng trong 2–4 tuần đầu nếu có tiền sử sẹo lồi hoặc da nhạy cảm, '
            'vì lòng trắng chứa albumin có thể kích ứng trong giai đoạn lành thương sớm. '
            'Thời điểm: từ ngày thứ 3–5 sau mổ (khi nhu động ruột hồi phục, không còn đầy hơi) '
            'có thể bắt đầu ăn trứng; tuần đầu ăn lòng đỏ, từ tuần 2 ăn cả quả. '
            'Số lượng: 1–2 quả/ngày. '
            'Cách chế biến: luộc chín hoặc nấu cháo là tốt nhất; tránh chiên, ốp la, trứng sống '
            'vì dầu mỡ nhiều và vi khuẩn nguy cơ. '
            'Thực phẩm kiêng thực sự sau sinh mổ: đồ sống, cua ốc (hàn, cản lành thương), '
            'gạo nếp (dễ mưng mủ), rượu bia, thức ăn cay nóng và đồ chiên rán (nên chờ 4–6 tuần).'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s4_052: quả tầm bóp phơi khô nấu nước
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s4_052',
        'question': 'Quả tầm bóp phơi khô nấu nước uống có thật sự tốt cho sức khỏe không?',
        'url': 'https://www.vinmec.com/vie/bai-viet/cay-tam-bop-co-tac-dung-gi-vi',
        'text': (
            'Cây tầm bóp (Physalis angulata) — công dụng và lưu ý y khoa: '
            'Tầm bóp còn gọi là bôm bốp, thù lù cạnh, lồng đèn; thân thảo mọc dại, có nguồn gốc từ châu Mỹ nhiệt đới. '
            'Thành phần hóa học: Physalin A–D, Physagulin A–G, alkaloid, vitamin C, vitamin A, '
            'chất xơ, protein, sắt, kẽm, canxi, photpho, magie. '
            'Tính vị: toàn cây đắng, tính mát; riêng quả tính bình, vị chua nhẹ — không có độc tính. '
            'Công dụng được ghi nhận: '
            '(1) Thanh nhiệt, giải độc, hỗ trợ mát gan — dùng toàn cây hoặc quả sắc nước uống như trà; '
            '(2) Kháng viêm, hỗ trợ điều trị gout và đái tháo đường (cơ chế kích thích tiết insulin sơ bộ); '
            '(3) Tăng đề kháng, hạ sốt nhờ vitamin C; '
            '(4) Hỗ trợ thị lực nhờ vitamin A; '
            '(5) Hoạt tính ức chế tế bào ác tính in vitro (vẫn đang nghiên cứu trên người). '
            'Cách dùng phơi khô nấu nước: dùng toàn cây (rễ, thân, lá, quả) phơi khô, '
            'sắc 15–20 g với 500 ml nước, uống như trà thanh nhiệt hàng ngày. '
            'Lưu ý: '
            '— Tác dụng điều trị bệnh lý cụ thể chưa được chứng minh đầy đủ trên người bằng thử nghiệm lâm sàng; '
            '— Chỉ nên dùng ngắn hạn, đúng liều, không thay thế chẩn đoán và điều trị y khoa; '
            '— Phụ nữ mang thai và trẻ em cần tham khảo bác sĩ trước khi dùng; '
            '— Tránh dùng đồng thời với thuốc tây nếu không có chỉ định.'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_034: ăn bao nhiêu calo mỗi ngày
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_034',
        'question': 'Người trưởng thành cần ăn bao nhiêu calo mỗi ngày và cách tính như thế nào?',
        'url': 'https://www.vinmec.com/vie/bai-viet/nhu-cau-calo-uoc-tinh-moi-ngay-theo-do-tuoi-gioi-tinh-vi',
        'text': (
            'Nhu cầu calo hàng ngày của người trưởng thành: '
            'Không có con số chung cho tất cả — nhu cầu phụ thuộc vào giới tính, tuổi, cân nặng, chiều cao và mức độ vận động. '
            'Bảng nhu cầu ước tính theo tuổi và giới tính: '
            'Nam 19–30 tuổi: 2.400–2.800 kcal/ngày; Nam 31–50 tuổi: 2.200–2.600 kcal/ngày; Nam trên 51 tuổi: 2.000–2.400 kcal/ngày. '
            'Nữ 19–30 tuổi: 1.800–2.200 kcal/ngày; Nữ 31–50 tuổi: 1.800–2.000 kcal/ngày; Nữ trên 51 tuổi: 1.600–1.800 kcal/ngày. '
            'Điều chỉnh theo mức vận động: '
            'Ít vận động (ngồi nhiều): lấy mức thấp trong khoảng; '
            'Vận động vừa (đi bộ 30–60 phút/ngày): tăng 300–500 kcal; '
            'Vận động mạnh (tập thể thao cường độ cao): tăng thêm 500–700 kcal. '
            'Theo phân loại lao động Việt Nam: lao động nhẹ 2.200–2.400 kcal; lao động vừa 2.600–2.800 kcal; '
            'lao động nặng 3.000–3.600 kcal; lao động rất nặng > 3.600 kcal. '
            'Công thức Mifflin–St Jeor (tính BMR trước, sau đó nhân hệ số hoạt động): '
            'Nữ: BMR = 10 × cân nặng (kg) + 6,25 × chiều cao (cm) – 5 × tuổi – 161; '
            'Nam: BMR = 10 × cân nặng (kg) + 6,25 × chiều cao (cm) – 5 × tuổi + 5. '
            'Hệ số hoạt động: ít vận động × 1,2; vận động nhẹ × 1,375; vận động vừa × 1,55; vận động nhiều × 1,725. '
            'Lưu ý: sau 30 tuổi, BMR giảm khoảng 2–3% mỗi thập kỷ; '
            'mang thai tăng thêm 300–500 kcal/ngày; cho con bú tăng thêm 400–500 kcal/ngày; '
            'người muốn giảm cân nên giảm 500 kcal/ngày so với nhu cầu (không xuống dưới 1.200 kcal với nữ và 1.500 kcal với nam).'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_038: sau sinh 2 tuần ăn đồ chiên rán
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_038',
        'question': 'Sau sinh 2 tuần có ăn đồ chiên rán được không, ảnh hưởng đến sữa mẹ thế nào?',
        'url': 'https://www.vinmec.com/vie/bai-viet/san-phu-sinh-mo-nen-kieng-nhung-mon-nao-vi',
        'text': (
            'Đồ chiên rán sau sinh — thời điểm an toàn và ảnh hưởng đến sữa mẹ: '
            'Sau sinh (cả thường và mổ), khuyến cáo chung là nên chờ ít nhất 4–6 tuần trước khi ăn lại đồ chiên rán. '
            'Lý do cần kiêng giai đoạn đầu: '
            '(1) Đồ chiên rán chứa nhiều chất béo bão hòa và trans — khó tiêu, dễ gây đầy bụng, táo bón '
            'trong khi hệ tiêu hóa chưa hồi phục hoàn toàn sau sinh; '
            '(2) Ảnh hưởng đến sữa mẹ: chế độ ăn nhiều dầu mỡ làm thay đổi thành phần lipid trong sữa, '
            'có thể gây đầy hơi, rối loạn tiêu hóa ở trẻ bú mẹ; '
            '(3) Đồ chiên ở nhiệt độ cao tạo acrylamide và các hợp chất oxy hóa, có thể làm chậm lành vết thương; '
            '(4) Dễ gây tăng cân không kiểm soát trong giai đoạn hậu sản. '
            'Nếu sau 6 tuần muốn ăn trở lại: bắt đầu từ lượng nhỏ (vài miếng), '
            'dùng dầu thực vật lành mạnh (dầu ô-liu, dầu cải), chiên ít lần, không tái sử dụng dầu; '
            'quan sát phản ứng của cơ thể và bé trong 24–48 giờ. '
            'Nếu sinh mổ (như trường hợp hỏi): nên chờ 6–8 tuần vì vết mổ cần thêm thời gian lành. '
            'Thực phẩm nên ưu tiên trong 4–6 tuần đầu: cháo, súp, cơm mềm, thịt luộc/hầm, cá hấp, '
            'rau nấu chín, trứng luộc — giàu protein, ít dầu mỡ, dễ tiêu hóa.'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_078: teo cơ chân nên ăn gì
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_078',
        'question': 'Bị teo cơ chân nên ăn gì để cải thiện, chỉ ăn uống có đủ không?',
        'url': 'https://www.vinmec.com/vie/bai-viet/bi-teo-co-chan-phai-lam-sao-vi',
        'text': (
            'Dinh dưỡng cho người bị teo cơ chân — và giới hạn của chỉ chế độ ăn: '
            'Teo cơ (suy giảm khối lượng và sức mạnh cơ) xảy ra do ít vận động, chấn thương, '
            'bệnh thần kinh–cơ, suy dinh dưỡng hoặc bệnh mạn tính. '
            'Dinh dưỡng cần thiết: '
            '(1) Protein chất lượng cao — axit amin thiết yếu để tái tạo sợi cơ: '
            'nhu cầu tăng lên 1,2–1,6 g/kg cân nặng/ngày (thông thường 0,8 g/kg); '
            'ưu tiên ức gà, cá (đặc biệt cá hồi, cá ngừ), thịt bò nạc, trứng, sữa, đậu hũ, đậu đen, đậu lăng. '
            '(2) Omega-3 — giảm viêm, hỗ trợ phục hồi mô cơ, cải thiện dẫn truyền thần kinh: '
            'cá hồi, cá thu, hạt lanh, hạt chia, óc chó. '
            '(3) Vitamin D — thiết yếu cho chức năng cơ và thần kinh: '
            'cá, trứng, sữa tăng cường; phơi nắng 10–15 phút, 3 lần/tuần. '
            '(4) Canxi — phối hợp với vitamin D duy trì co bóp cơ bình thường: '
            'sữa, sữa chua, rau lá xanh đậm (cải xanh, bông cải), tôm tép nhỏ ăn cả vỏ. '
            '(5) Vitamin B12 và sắt — hỗ trợ chức năng thần kinh và vận chuyển oxy đến cơ. '
            'Giới hạn của chế độ ăn đơn thuần: '
            'Dinh dưỡng là nền tảng nhưng phục hồi thực sự chỉ xảy ra khi kết hợp với vận động và điều trị nguyên nhân. '
            'Cần vật lý trị liệu (đạp xe tại chỗ, co duỗi chân, bài tập thụ động) để kích thích cơ phát triển trở lại. '
            'Nếu teo cơ kéo dài > 2–3 tuần không rõ lý do, cần khám chuyên khoa cơ–xương–khớp hoặc thần kinh '
            'để chẩn đoán nguyên nhân (bệnh thần kinh ngoại biên, tổn thương tủy, v.v.).'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_080: ho mệt ngán ăn nên ăn gì, có ăn trứng được không
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_080',
        'question': 'Ho mệt ngán ăn có ăn trứng gà được không, hay kiêng vì sợ sinh đàm?',
        'url': 'https://www.vinmec.com/vie/bai-viet/co-can-kieng-thit-ga-tom-khi-bi-ho-vi',
        'text': (
            'Ho có ăn trứng gà được không — thực hư quan niệm sinh đàm: '
            'Y học hiện đại không có bằng chứng chứng minh trứng gà làm tăng đàm hoặc làm ho nặng hơn. '
            'Quan niệm "trứng sinh đàm" là kinh nghiệm dân gian, chưa được kiểm chứng khoa học. '
            'Ngược lại, khi bị ho mệt và ngán ăn, trứng gà là thực phẩm rất phù hợp vì: '
            '(1) Giàu protein chất lượng cao — cung cấp axit amin để sửa chữa niêm mạc đường hô hấp bị tổn thương; '
            '(2) Giàu vitamin A (tăng cường niêm mạc), D (miễn dịch), B12, kẽm và selen — '
            'tất cả đều hỗ trợ hệ miễn dịch chống nhiễm trùng; '
            '(3) Mềm, dễ nuốt, dễ tiêu — phù hợp khi ngán ăn và cổ họng khó chịu. '
            'Cách chế biến phù hợp khi ho: '
            'Luộc chín, nấu cháo trứng, trứng hấp — tránh chiên ốp la hoặc trứng sống '
            'vì dầu mỡ có thể kích ứng cổ họng. '
            'Trường hợp cần thận trọng: dị ứng trứng (thay bằng cá, đậu hũ); '
            'ho dị ứng/hen — tránh lòng trắng trứng vì có thể là dị nguyên. '
            'Thực phẩm thực sự nên tránh khi ho: đồ cay (ớt, tiêu — kích ứng cổ họng), '
            'đồ lạnh (kem, nước đá — co thắt phế quản), rượu bia, đồ chiên nhiều dầu. '
            'Gợi ý thực đơn khi ho mệt ngán ăn: cháo trứng, súp gà, cháo cá — '
            'vừa cung cấp năng lượng, vừa dễ ăn, vừa hỗ trợ phục hồi.'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_129: cắt túi mật uống sữa đầy hơi tiêu chảy
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_129',
        'question': 'Sau cắt túi mật uống sữa bị đầy hơi tiêu chảy, loại sữa nào an toàn?',
        'url': 'https://www.vinmec.com/vie/bai-viet/nhung-dieu-can-biet-ve-che-do-an-sau-khi-cat-tui-mat',
        'text': (
            'Chế độ ăn uống sau cắt túi mật — vấn đề sữa và chất béo: '
            'Sau phẫu thuật cắt túi mật (cholecystectomy), mật không còn được dự trữ và tiết ra từng lúc mà '
            'chảy thẳng liên tục từ gan vào ruột. Điều này làm giảm khả năng nhũ hóa chất béo, '
            'gây tiêu chảy (đặc biệt 2–4 tuần đầu), đầy hơi và khó tiêu khi ăn thực phẩm nhiều mỡ. '
            'Tại sao sữa nguyên kem gây vấn đề: sữa nguyên kem chứa 3,5%+ chất béo — '
            'hệ tiêu hóa thiếu mật dự trữ không xử lý được, dẫn đến đầy hơi, chuột rút, tiêu chảy. '
            'Loại sữa an toàn sau cắt túi mật: '
            '(1) Sữa tách béo/sữa gầy (hàm lượng béo < 1%) — lựa chọn tốt nhất giai đoạn đầu; '
            '(2) Sữa ít béo (1–1,5% béo) — dung nạp tốt khi đã qua 4–6 tuần; '
            '(3) Sữa thực vật: sữa đậu nành, sữa hạnh nhân, sữa gạo — ít béo, dễ tiêu; '
            '(4) Sữa chua ít béo — men vi sinh có thể hỗ trợ tiêu hóa; '
            '(5) Phô mai ít béo (ricotta, cottage cheese). '
            'Nguyên tắc chung về chất béo: không quá 30% tổng calo từ chất béo; '
            'tránh bơ, mỡ động vật, kem, thịt mỡ, đồ chiên rán — đặc biệt trong 4–6 tuần đầu. '
            'Lưu ý thực tế: '
            'Tiêu chảy sau cắt túi mật thường tự cải thiện sau 4–8 tuần khi ruột thích nghi. '
            'Nếu triệu chứng nặng (đau quặn, tiêu chảy > 3 lần/ngày, sụt cân), cần tái khám bác sĩ tiêu hóa. '
            'Cắt túi mật không có nghĩa là kiêng sữa suốt đời — hầu hết bệnh nhân dung nạp tốt '
            'sữa tách béo hoặc sữa thực vật sau khi cơ thể thích nghi.'
        ),
    },
    # ------------------------------------------------------------------
    # thucuc_s5_161: lá mơ chữa đầy bụng tiêu chảy đau dạ dày
    # ------------------------------------------------------------------
    {
        'qid': 'thucuc_s5_161',
        'question': 'Lá mơ lông chữa đầy bụng tiêu chảy đau dạ dày — dùng lâu có hại không, kỵ thuốc gì?',
        'url': 'https://www.vinmec.com/vie/bai-viet/tac-dung-cua-la-mo-long-vi',
        'text': (
            'Lá mơ lông (Paederia tomentosa) — tác dụng, cách dùng và lưu ý an toàn: '
            'Thành phần hoạt chất: alkaloid paederin, tinh dầu sulfur dimethyl disulphide, '
            'flavonoid — có tính kháng sinh và kháng viêm. '
            'Tác dụng được ghi nhận trong y học cổ truyền và nghiên cứu sơ bộ: '
            '(1) Tiêu hóa: kháng khuẩn với E. coli, Shigella (nguyên nhân tiêu chảy, lỵ); '
            'giảm co thắt cơ trơn, dịu cơn đau quặn bụng; kích thích tiết dịch vị, giảm đầy hơi khó tiêu; '
            'hỗ trợ điều trị kiết lỵ, viêm đại tràng co thắt. '
            '(2) Dạ dày: giảm đau dạ dày, kích thích ăn ngon. '
            '(3) Khác: trị thấp khớp, cam tích trẻ em, tổn thương da dân gian. '
            'Cách dùng thông thường: '
            '— Tươi: 20–30 g lá tươi rửa sạch, giã lấy nước uống ngay hoặc hấp chín với trứng; '
            '— Nước sắc: 15–20 g lá khô sắc với 300 ml nước, uống 2 lần/ngày. '
            'Tác dụng phụ và giới hạn liều: '
            '— Uống quá 30–50 g lá tươi/ngày: buồn nôn, đau bụng, nôn; '
            '— Uống khi bụng đói có thể kích ứng niêm mạc dạ dày. '
            'Tương tác thuốc: lá mơ có thể làm giảm hấp thu kháng sinh đường ruột và thuốc tây trị tiêu chảy '
            'nếu uống cùng lúc — nên cách 2 giờ. '
            'Đối tượng cần thận trọng hoặc tránh dùng: '
            '— Người thể hàn (tay chân lạnh, hay mệt, phân lỏng) — lá mơ tính mát có thể làm nặng thêm; '
            '— Phụ nữ mang thai 3 tháng đầu — thiếu dữ liệu an toàn, tránh dùng liều cao; '
            '— Dùng lâu dài (> 4 tuần liên tục) không được khuyến khích vì chưa có dữ liệu an toàn dài hạn. '
            'Kết luận: lá mơ có tác dụng hỗ trợ tiêu hóa có cơ sở, nhưng '
            'không thay thế điều trị y khoa cho viêm đại tràng, nhiễm khuẩn nặng hay loét dạ dày.'
        ),
    },
]

with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
    for e in entries:
        qid = e['qid']
        url = e['url']
        text = e['text']
        row = {
            'chunk_id': make_id(f'missing9_{qid}_{url}'),
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

print(f'\nDone — {len(entries)} chunks appended to corpus.')
