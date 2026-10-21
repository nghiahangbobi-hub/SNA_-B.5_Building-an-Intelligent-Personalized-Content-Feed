# BÁO CÁO PHẦN VIỆC CÁ NHÂN — PHUC-01 ĐẾN PHUC-05

**Đề tài:** B.5 – Building an Intelligent Personalized Content Feed  
**Bộ dữ liệu:** MIND-small  
**Repository nguồn:** https://github.com/Nobita534/mind-temporal-eval-baselines  
**Phạm vi kết quả:** Thực nghiệm offline, seed 0; đối chiếu từ các artifacts trong repository ngày 20/10/2026 và lần pytest người thực hiện chạy lại ngày 21/10/2026.

## 1. Task đã thực hiện

| Ticket | Nội dung | Sản phẩm chính |
|---|---|---|
| **PHUC-01** | Xây dựng Evaluator chuẩn hóa: schema, AUC, MRR, nDCG@5, nDCG@10, đánh giá theo impression và phân nhóm. | Bộ đánh giá dùng chung cho các mô hình. |
| **PHUC-02** | Xây dựng B1 Random và B2 Popularity (Cumulative/24h), bảo đảm chỉ sử dụng các positive labels trong quá khứ được phép truy cập. | Baselines làm mốc so sánh. |
| **PHUC-03** | Xây dựng B3 Category Matching và B4 Semantic Matching với frozen MiniLM embeddings. | Baselines có tính cá nhân hóa. |
| **PHUC-04** | Thực nghiệm E1 so sánh B2 Cumulative theo GTS Clean, Random Split và GTS Static. | Đánh giá mức tăng metric khi sử dụng thông tin không hợp lệ. |
| **PHUC-05** | Regression tests, synthetic smoke tests, score replay, adapter cho predictions bên ngoài và tổng hợp artifacts. | Khả năng tích hợp, kiểm chứng và truy vết kết quả. |

## 2. Mô tả các file quan trọng

| Nhóm | File | Vai trò |
|---|---|---|
| `src/data_loader/` | `mind_reader.py`, `schema.py` | Đọc MIND (`behaviors.tsv`, `news.tsv`); chuẩn hóa impression, candidate, label, History và prediction. |
| | `temporal_split.py`, `random_split.py` | Chia dữ liệu theo thời gian cho benchmark chính, chia 80/10/10 cho E1 Random. |
| | `prediction_reader.py` | Đọc/kiểm tra score CSV của B5, B6, THG-Rec trước khi chấm. |
| `src/baselines/` | `base.py`, `b1_random.py`, `b2_popularity.py` | Interface scoring chung; Random và Popularity với cutoff. |
| | `b3_category.py`, `b4_semantic_plm.py` | Chấm điểm bằng lịch sử chủ đề và độ tương đồng embeddings. |
| | `e1_popularity.py` | Biến thể Popularity dùng trong ba protocol E1. |
| `src/evaluation/` | `metrics.py`, `evaluator.py`, `group_analyzer.py` | Tính ranking metrics, macro-average và group-wise metrics. |
| | `leakage_comparison.py`, `handoff.py` | So sánh protocol, tổng hợp/kiểm tra artifacts thực nghiệm. |
| `scripts/` | `run_baselines.py`, `run_e1_leakage.py` | Chạy thực nghiệm B1–B4 và E1. |
| | `cache_plm_embeddings.py` | Tạo, lưu và kiểm tra MiniLM embeddings cho B4. |
| | `evaluate_predictions.py`, `build_handoff.py` | Đánh giá scores bên ngoài và tổng hợp kết quả seed 0. |
| `configs/` | `eval_config.yaml`, `baselines_config.yaml`, `content_baselines_config.yaml`, `e1_protocols.yaml` | Lưu quy tắc đánh giá, scoring, PLM và E1. |
| `tests/` | Bảy file `test_*.py` | Kiểm tra từng thành phần và toàn pipeline. |

**Không tính là kết quả đã hoàn thành:** `src/evaluation/stats_test.py` là placeholder cho phần kiểm định thống kê trong tương lai.

## 3. Workflow

```text
MIND-small (behaviors.tsv + news.tsv)
    ↓
Đọc dữ liệu, lấy impression / candidate / History / labels
    ↓
Global Temporal Split (GTS)
    ├── Train: 09–13/11/2019 — 126.695 impressions
    ├── Validation: 14/11/2019 — 30.270 impressions
    └── Test: 15/11/2019 — 73.152 impressions (official dev)
    ↓
Chấm điểm từng candidate bằng B1 / B2 / B3 / B4
    ↓
Ghép scores với ground-truth labels để đánh giá
    ↓
Shared Evaluator → AUC / MRR / nDCG@5 / nDCG@10
    ↓
Overall metrics + Group-wise metrics + manifests
```

- **B1:** Tạo score giả ngẫu nhiên theo seed và định danh impression/candidate; không học từ Train.
- **B2:** Dùng số positive-label events từ các impressions quá khứ; Validation sử dụng Train 09–13/11, Test sử dụng Train+Validation 09–14/11. Cutoff là **00:00 đầu ngày**; bản 24h dùng `[cutoff−24h, cutoff)`.
- **B3:** Ước lượng sở thích theo category/subcategory của các news IDs trong **History của impression hiện tại**.
- **B4:** Biểu diễn title/abstract bằng frozen `all-MiniLM-L6-v2`, trung bình các embeddings trong History và tính cosine similarity với từng candidate.
- **E1:** Chạy B2 Cumulative trong ba protocol khác nhau để so sánh kết quả khi nguồn đếm và/hoặc cách chia dữ liệu thay đổi.
- **PHUC-05:** Kiểm tra schema CSV dự đoán của mô hình bên ngoài, dựng lại ground truth từ MIND và đánh giá bằng cùng Evaluator.

## 4. Các test đã viết

| Test module | Kiểm tra chính |
|---|---|
| `test_metrics_edge_cases.py` | Công thức metric, nhiều positive, tie scores, labels không hợp lệ, giá trị undefined, macro-average và nhóm dữ liệu. |
| `test_temporal_leakage.py` | Đọc dữ liệu, Temporal Split, cutoff, click index, chống sử dụng click tương lai trong B2, tính ổn định của B1. |
| `test_content_baselines.py` | B3 category/subcategory fallback, B4 cosine/mean History, trường hợp thiếu dữ liệu, tính toàn vẹn của embedding cache. |
| `test_e1_leakage.py` | Random Split xác định theo seed, ba protocol E1, nguồn đếm không hợp lệ, đối chiếu E1 Clean với PHUC-02. |
| `test_prediction_reader.py` | Kiểm tra candidate order, độ bao phủ impressions, scores sai định dạng/NaN, provenance và CSV đầu vào. |
| `test_handoff.py` | Kiểm tra đủ metrics, nhất quán giữa manifests, inputs, seed, protocol và từ chối artifacts không hợp lệ. |
| `test_smoke_pipeline.py` | Chạy pipeline hai lần bằng **synthetic data** để xác nhận đầu ra nhất quán. |

**Kết quả kiểm thử mới nhất do người thực hiện cung cấp:** `199 passed, 42 subtests passed in 35.51s`, chạy bằng:

```powershell
python -m pytest tests/ -q -p no:cacheprovider --basetemp "$safeTemp\pytest-run"
```

Đường dẫn temp tùy máy. Lần chạy đầu bị `PermissionError: [WinError 5]` tại thư mục temp mặc định của Windows; sau khi chỉ định thư mục temp có quyền truy cập, toàn bộ suite đã pass. Kết quả này xác nhận **repo cá nhân**, không tự động xác nhận repo nhóm.

Ngoài unit tests, log PHUC-05 ghi nhận replay 8.615 candidate scores trên 40 impressions duy nhất (5 cấu hình mô hình), và việc đánh giá lại B2 Cumulative trên 30.270 Validation + 73.152 Test impressions cho kết quả giống artifacts ban đầu.

## 5. Result và giải thích

### 5.1. B1–B4 trên Test (GTS Clean, seed 0)

| Baseline | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| B1 Random | 0,499228 | 0,217864 | 0,222431 | 0,285491 |
| B2 Cumulative | 0,531752 | 0,238201 | 0,246042 | 0,309817 |
| B2 24h | 0,535480 | 0,239505 | 0,247976 | 0,312459 |
| B3 Category | 0,604632 | 0,293689 | 0,319552 | 0,378695 |
| **B4 Semantic PLM** | **0,634951** | **0,307913** | **0,336576** | **0,395581** |

**Diễn giải:** B1 là mốc ngẫu nhiên. B2 cho thấy số click quá khứ có giá trị dự đoán; ở lần chạy này cửa sổ 24h nhỉnh hơn Cumulative. B3 vượt B2 khi tận dụng History theo chủ đề. B4 có nDCG@10 cao nhất, gợi ý rằng độ tương đồng ngữ nghĩa giàu thông tin hơn cách ghép category trong phạm vi thực nghiệm này. Đây là chênh lệch **quan sát offline với seed 0**, chưa có kiểm định ý nghĩa thống kê.

### 5.2. E1 — Ảnh hưởng của data leakage trên B2 Cumulative

| Protocol | Số Test impressions | AUC | nDCG@10 |
|---|---:|---:|---:|
| GTS Clean | 73.152 | 0,531752 | 0,309817 |
| Random 80/10/10 | 23.013 | 0,600933 | 0,374188 |
| GTS Static | 73.152 | 0,599364 | 0,372301 |

GTS Static dùng dữ liệu 09–15/11, **bao gồm chính labels của tập đánh giá**. Trên cùng 73.152 Test impressions, nDCG@10 tăng từ 0,309817 lên 0,372301 (**+0,062485 điểm; +20,17% tương đối**). Đây là **metric inflation do protocol cố ý không hợp lệ**, không phải bằng chứng mô hình dự đoán tốt hơn trên dữ liệu tương lai.

Random cũng cho metrics cao hơn nhưng **thay đổi tập Test** và nguồn dữ liệu đếm; không thể diễn giải chênh lệch của Random thành tác động thuần của leakage.

**Artifacts chính:** `outputs/tables/phuc05_handoff_seed0_2026-10-20/{metrics.csv,results.md,manifest.json}` và `outputs/tables/phuc04_e1_mindsmall_seed0_2026-10-20/inflation.csv` trong repo cá nhân. Khi chuyển sang repo nhóm, lưu bản tóm tắt tại `results/phuc/`.

## 6. Giới hạn và liên kết tương lai

- **Thời gian sự kiện:** MIND không cung cấp timestamp riêng cho từng click; B2 dùng timestamp của impression làm proxy cho positive event. Timestamps trong History không có để kiểm chứng chi tiết giả định “History trước impression”.
- **Publication time:** `news.tsv` không có thời điểm xuất bản đáng tin cậy cho việc chứng minh article availability tại một cutoff lịch sử.
- **PLM hồi cứu:** Frozen MiniLM dùng trong B4 chưa được chứng minh sẵn có cùng weights/data tại năm 2019; phù hợp mô tả là retrospective semantic baseline, không khẳng định triển khai lịch sử thật.
- **Đánh giá offline:** Ranking metrics không tự chứng minh tăng CTR hoặc hiệu quả A/B testing trong sản phẩm.
- **Thống kê:** Phạm vi hiện tại là seed 0, chưa có 5 seeds, paired significance tests và hiệu chỉnh Holm theo hướng nghiên cứu tiếp theo.
- **E1 chưa bao quát hết:** Thí nghiệm E1 chỉ triển khai B2 Cumulative; chưa kiểm chứng leakage thực tế trong B5/B6/THG-Rec.
- **Tái lập trên máy khác:** Raw MIND, per-impression predictions, embedding cache và split membership lớn không nằm trong các file source kết quả gọn để đưa lên Git; cần chuẩn bị riêng nếu chạy lại toàn bộ benchmark.

**Liên kết:** B5, B6 và THG-Rec có thể xuất scores cho toàn bộ candidates rồi dùng `scripts/evaluate_predictions.py` để được đánh giá thống nhất. Kết quả B1–B4 làm mốc so sánh; group metrics phục vụ phân tích theo History/cold-warm/category; E1 mở đường cho việc rà soát temporal leakage ở các mô hình phức tạp. Các đánh giá tiếp theo cần chuẩn hóa protocol và bổ sung thống kê nhiều seed trước khi đưa ra kết luận tổng quát.

**Kết luận:** PHUC-01–05 tạo thành bộ khung **Temporal Recommendation Evaluation Pipeline**, gồm đọc dữ liệu, scoring baselines, evaluation, thí nghiệm leakage và kiểm chứng kết quả. Đóng góp trọng tâm là cơ sở đánh giá chung, chứ không chỉ một baseline có điểm số cao.
