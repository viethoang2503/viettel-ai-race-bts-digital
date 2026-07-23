# Thiết kế: Đẩy Score baseline (~0.62) lên gần 0.80 trong 7 ngày còn lại

## 1. Bối cảnh & mục tiêu

Plan 1 (baseline core pipeline) đã hoàn thành: cả 7 scene (`HCM0421`, `HCM0539`, `HCM0540`,
`HCM0644`, `HCM0674`, `chair`, `bonsai`) đã train, render, đóng gói thành `submission.zip` hợp
lệ. Điểm holdout tự đo hiện tại (thang 0-1, ước tính tương ứng thang hiển thị 0-100 trên BXH):

| Scene | Score | LPIPS | SSIM | PSNR |
|---|---|---|---|---|
| HCM0421 | 0.6716 | - | - | - |
| HCM0539 | 0.7069 | - | - | - |
| HCM0540 | 0.6197 | - | - | - |
| HCM0644 | 0.6147 | - | - | - |
| HCM0674 | 0.6945 | - | - | - |
| chair | 0.5559 | 0.4448 | 0.5355 | 17.31 |
| bonsai | 0.4922 | 0.5615 | 0.5676 | 14.65 |
| **Trung bình** | **≈0.622** | | | |

Đây là baseline 3DGS thuần, không kỹ thuật nâng cao nào. Mục tiêu giai đoạn này: đẩy trung bình
lên gần **0.80** trước deadline **30/07/2026** (còn 7 ngày kể từ hôm viết spec, 23/07/2026).

`chair` và `bonsai` là 2 scene thấp nhất, kéo trung bình xuống nhiều nhất — ưu tiên xử lý sâu
hơn 5 scene HCM.

## 2. Ràng buộc

- **Ngân sách GPU: không còn là rào cản** — người dùng xác nhận có thể mua thêm unit Colab Pro.
  Rào cản thật là **thời gian lịch (7 ngày)**, không phải tiền.
- **Nhân lực:** người dùng dành được 6-10 tiếng/ngày để canh Colab (mount session, restart khi
  disconnect, theo dõi training) — đủ để chạy nhiều lượt train tuần tự mỗi ngày, nhưng vẫn hữu
  hạn.
- **GPU tier: giữ nguyên L4** (theo quyết định người dùng) — không chuyển A100, tránh rủi ro gặp
  lỗi mới phải debug lại đúng lúc gấp deadline.
- **Chạy tuần tự, không giả định song song** — Colab Pro (không phải Pro+) không đảm bảo nhiều
  session GPU đồng thời ổn định. Lập kế hoạch coi 1 session tại 1 thời điểm là kịch bản cơ sở;
  chạy song song (nếu thử và thành công) chỉ là phần thưởng thêm, không phải điều kiện tiên
  quyết.
- Vẫn phải tuân thủ mọi ràng buộc đã có ở
  `docs/superpowers/specs/2026-07-18-nvs-bts-pipeline-design.md` (không dùng dữ liệu ngoài,
  không truy xuất ground-truth test thật, phải tái lập được, không chỉnh sửa ảnh thủ công).

## 3. Quan hệ với Plan 2 đã thiết kế sẵn

`docs/superpowers/plans/2026-07-18-advanced-techniques.md` đã thiết kế đầy đủ 11 task, kèm code
thật + test cho phần thuần Python, cho: VRAM guard (Task 1), floater/background prune (Task 2),
sparse depth target (Task 3), appearance embedding (Task 4), depth reg loss (Task 5), auto-select
theo holdout (Task 6), anti-aliasing qua flag `antialiasing` có sẵn trong baseline gốc (Task 7),
variant-aware training loop chạy 5 biến thể `baseline/depth_reg/anti_alias/appearance_embed/
full_stack` (Task 8), orchestrator chạy full ma trận biến thể (Task 9,
`run_experiment_matrix_pipeline`), reproducibility bundle (Task 10), visual QA (Task 11 — **đã
code và đang dùng ở Bước 7 hai notebook**).

**Spec này không viết lại Task 1-11** — chúng vẫn là thiết kế đúng, chưa từng bị bug hay sai sót
kỹ thuật nào được phát hiện. Spec này chỉ định nghĩa **những gì CẦN THÊM** để việc thực thi 11
task đó thực sự khả thi trong 7 ngày và nhắm đúng vào 2 scene yếu nhất:

1. Một bước **chẩn đoán** trước khi train thêm bất kỳ gì (mới — chưa có trong Task 1-11).
2. Một quy ước **rút gọn iteration khi screening** để ma trận 5 biến thể × 7 scene kịp thời gian
   (điều chỉnh cách gọi Task 8/9, không đổi code Task 8/9).
3. Một **bounded hyperparameter search** riêng cho `bonsai`+`chair`, nằm ngoài phạm vi ban đầu
   của Task 6/9 (`select_best_candidate` vốn chỉ so 5 biến thể — sẽ tái dùng y nguyên hàm này,
   chỉ mở rộng danh sách `candidates` truyền vào).
4. Trình tự lắp ráp cuối cùng (full-data retrain -> blind render -> QA -> reproducibility bundle
   -> merge) — nối các mảnh đã có (`run_baseline_pipeline`, `package_submission`,
   `validate_submission`, Bước 8 hiện tại của `colab_runner_hcm.ipynb`) theo đúng thứ tự mới.

## 4. Giai đoạn 0 — Chẩn đoán (mới, làm trước tiên, không tốn GPU)

**Vấn đề cần trả lời:** điểm thấp của `bonsai`/`chair` là do baseline vốn khó (extrapolation +
video cầm tay rung), hay có lỗi cụ thể (pose sai, distortion chưa xử lý, exposure lệch nặng,
floater cụ thể)? Trả lời sai câu này sẽ khiến các bước sau tốn GPU vào sai hướng.

**Vì holdout là ảnh tách ra từ tập train (có ground-truth thật)**, khác với `test_poses.csv`
thật (không có GT) — có thể so sánh trực tiếp ảnh render vs ảnh gốc, không chỉ nhìn ảnh render
đơn lẻ như `find_suspicious_renders` (Task 11) đang làm.

**Thành phần mới:**
- `src/diagnostics/scene_diagnosis.py` — hàm `rank_holdout_by_score(per_image_metrics: dict[str, dict]) -> list[tuple[str, float]]`
  (thuần Python, có test): nhận `{image_name: {"lpips":.., "ssim":.., "psnr":..}}`, trả về danh
  sách tên ảnh sắp xếp theo Score tăng dần (tệ nhất trước).
- Điều chỉnh nhỏ ở `run_baseline_pipeline` (`src/orchestrator/run_pipeline.py`): hiện tại
  `compute_pair_metrics` đã tính đúng theo từng ảnh nhưng bị gộp trung bình ngay, không giữ lại
  per-image. Thêm field `per_image_metrics: dict[str, dict[str, dict[str, float]]]` (scene ->
  image_name -> {lpips, ssim, psnr}) vào `PipelineResult` để giữ lại — không đổi hành vi tính
  Score hiện có, chỉ giữ thêm dữ liệu đã tính sẵn.
- 1 cell notebook mới (chạy trên Colab, dùng dữ liệu holdout đã render sẵn ở Plan 1) hiển thị:
  với mỗi scene, 5 ảnh holdout tệ nhất theo `rank_holdout_by_score` — ảnh predicted cạnh ảnh
  ground-truth, kèm số LPIPS/SSIM/PSNR từng ảnh. Người dùng + tôi cùng xem, phân loại thủ công:
  mờ đều? floater cụ thể? sai màu/exposure? sai hình học/pose? Đây là bước quan sát bằng mắt có
  chủ đích, không tự động phân loại nguyên nhân (tự động hoá việc này không đáng tin với 7 ngày
  còn lại).

**Đầu ra giai đoạn này:** ghi chú ngắn cho từng scene (đặc biệt `bonsai`, `chair`) về nguyên
nhân nghi ngờ chính — quyết định trực tiếp candidate nào đáng thử ở Giai đoạn 2 (ví dụ: nếu
`bonsai` toàn bộ ảnh tệ đều lệch màu → ưu tiên `appearance_embed`; nếu tệ đều ở rìa quỹ đạo →
chấp nhận đây là giới hạn extrapolation, không phải bug, dồn lực vào depth_reg/floater thay vì
cố sửa "bug" không tồn tại).

## 5. Giai đoạn 1 — Chạy ma trận 5 biến thể đã thiết kế sẵn (Task 8/9), rút gọn iteration để screening

Thực thi đúng `run_experiment_matrix_pipeline` (Task 9) cho cả 7 scene, dùng `ALL_TRAINING_VARIANTS`
(Task 8) không đổi. Ước tính: 5 biến thể × 7 scene = 35 lượt train.

**Điều chỉnh duy nhất so với thiết kế gốc:** dùng `iterations` rút gọn (ví dụ 15000 thay vì
30000) cho vòng screening này — mục đích là **xếp hạng tương đối** giữa 5 biến thể, không phải
lấy checkpoint cuối để nộp bài. Rủi ro: thứ hạng ở 15k có thể không hoàn toàn khớp ở 30k — chấp
nhận rủi ro này ở vòng screening để tiết kiệm ~50% thời gian, vì đây chỉ là bước *chọn* biến thể,
không phải bước chốt. Nếu 2 biến thể đầu bảng của 1 scene sít sao (chênh lệch Score < 0.01),
chạy thêm 1 lượt so sánh ở 30k cho riêng 2 candidate đó trước khi chốt.

`select_best_candidate` (Task 6) chọn winner mỗi scene như thiết kế gốc, không đổi.

**Ước tính thời gian:** 35 lượt × ~35-45 phút (rút gọn iteration) ≈ 20-26 giờ GPU — nằm gọn
trong phạm vi ngày đầu + ngày thứ hai của 7 ngày.

## 6. Giai đoạn 2 — Bounded hyperparameter search sâu cho `bonsai` + `chair` (mới)

Chỉ áp dụng cho 2 scene yếu nhất, KHÔNG áp dụng full-grid cho cả 7 scene (lý do: không kịp thời
gian dù ngân sách không giới hạn — xem mục 7).

Trên nền variant thắng của mỗi scene (từ Giai đoạn 1), thử thêm **tối đa 4 candidate/scene**,
mỗi candidate đổi 1-2 siêu tham số so với candidate thắng, dựa trên phát hiện ở Giai đoạn 0.
Danh sách tham số được phép đổi (không làm full lưới tổ hợp, chỉ đổi có mục đích theo chẩn
đoán): `densify_grad_threshold`, `densify_until_iter`, `opacity_reset_interval`, tổng
`iterations` (thử 45000 nếu Score vẫn đang tăng ở 30000, chưa bão hoà).

Mỗi candidate là 1 dict tương thích thẳng với `select_best_candidate` (Task 6) đã có sẵn — không
cần sửa hàm đó, chỉ nối thêm các candidate mới vào cùng list trước khi gọi.

**Ước tính thời gian:** 2 scene × 4 candidate × ~1-1.5 giờ (chạy full iteration vì đây là vòng
chốt, không rút gọn) ≈ 8-12 giờ GPU.

## 7. Giai đoạn 3 — Chốt & lắp ráp submission cuối

1. Với mỗi scene trong 7 scene, lấy config thắng cuối cùng (từ Giai đoạn 1 hoặc 2).
2. Train lại trên 100% dữ liệu (không tách holdout) với config thắng — tái dùng cơ chế
   "eval training -> full training" đã có sẵn trong `run_baseline_pipeline`, chỉ đổi
   `train_fn`/tham số theo config thắng thay vì baseline cố định.
3. Render `test_poses.csv` thật cho cả 7 scene (mù, không đối chiếu được).
4. QA mắt thường (`find_suspicious_renders`, Task 11) trên toàn bộ output mới.
5. Đóng gói reproducibility bundle (Task 10) — ghi lại config thắng, log, checkpoint mỗi scene.
6. Ghép 7 scene thành `submission.zip` cuối (mở rộng logic Bước 8 hiện tại của
   `colab_runner_hcm.ipynb` — thay vì luôn lấy `test_render/` mặc định, lấy đúng thư mục output
   của config thắng mỗi scene).
7. Chừa tối thiểu 1 ngày đệm trước 30/07/2026 để xử lý phát sinh.

## 8. Ngoài phạm vi

- **Full hyperparameter grid (iterations × threshold × opacity-reset × SH-degree × LR-schedule)
  cho cả 7 scene** — không khả thi trong 7 ngày dù ngân sách không giới hạn (xem ước tính thời
  gian mục kế tiếp). Chỉ 2 scene yếu nhất được đào sâu (Giai đoạn 2).
- Chuyển sang A100 hoặc hạ tầng khác.
- Phụ thuộc vào chạy nhiều session Colab song song.
- Tự động phân loại nguyên nhân lỗi bằng thuật toán (Giai đoạn 0 chỉ hỗ trợ hiển thị, con người
  tự đánh giá).

## 9. Ước tính tổng thời gian GPU và đối chiếu ngân sách ngày

| Giai đoạn | Số lượt train | Thời gian/lượt | Tổng |
|---|---|---|---|
| 0 — Chẩn đoán | 0 (dùng lại output Plan 1) | - | ~0 |
| 1 — Ma trận 5 biến thể × 7 scene (15k iter) | 35 | ~35-45 phút | ~20-26 giờ |
| 2 — Bounded search bonsai+chair (30k iter) | 8 | ~1-1.5 giờ | ~8-12 giờ |
| 3 — Full-data retrain cuối × 7 scene | 7 | ~1-1.5 giờ | ~7-10 giờ |
| **Tổng** | 50 | | **~35-48 giờ** |

Ngân sách khả dụng: 7 ngày × 6-10 giờ/ngày = 42-70 giờ. **Vừa khít đến thoải mái**, còn dư biên
độ cho: session bị disconnect phải restart, thời gian render (nhanh hơn train nhiều nhưng vẫn
cộng dồn), thời gian con người xem QA/diagnosis. Nếu thực tế chạy chậm hơn ước tính, cắt giảm
trước tiên ở Giai đoạn 2 (giảm từ 4 xuống 2-3 candidate/scene) — không cắt Giai đoạn 1 hay 3 vì
đó là phần đảm bảo có bản nộp hợp lệ.

## 10. Kiểm thử

- Mọi phần thuần Python mới (`rank_holdout_by_score`, việc mở rộng `PipelineResult` với
  `per_image_metrics`, danh sách candidate mở rộng cho Giai đoạn 2) có `pytest` unit test chạy
  được trên máy local không GPU, theo đúng pattern Plan 1/Plan 2 đã dùng.
  `.venv/bin/python -m pytest -q` phải pass toàn bộ trước khi đưa lên Colab chạy thật.
- Phần cần GPU thật (Task 8/9 đã có sẵn, không đổi logic) verify trên Colab như Plan 1/2 đã làm.
- Giai đoạn 3 bước "train lại 100% dữ liệu" verify bằng cách xác nhận
  `validate_submission` không báo lỗi và `find_suspicious_renders` không phát hiện ảnh trắng bất
  thường, giống quy trình QA đã dùng ở Plan 1.

## 11. Rủi ro & phương án dự phòng

- **Rút gọn iteration ở Giai đoạn 1 làm sai thứ hạng biến thể:** giảm rủi ro bằng quy tắc "chênh
  lệch < 0.01 thì chạy lại ở full iteration" đã nêu ở mục 5.
- **Chẩn đoán (Giai đoạn 0) không tìm ra nguyên nhân rõ ràng cho bonsai/chair:** vẫn tiếp tục
  Giai đoạn 1/2 bình thường — chẩn đoán chỉ để ưu tiên candidate nào thử trước ở Giai đoạn 2, không
  phải điều kiện chặn tiến độ.
- **Depth reg làm giảm điểm ở scene có sparse depth nhiễu:** đã có cơ chế `select_best_candidate`
  tự loại — nếu biến thể `depth_reg`/`full_stack` thua `baseline` ở 1 scene, `baseline` (có thể
  cộng thêm floater cleanup) tự động được chọn, không cần can thiệp thủ công.
- **Hết thời gian trước deadline:** vì đã chừa 1 ngày đệm (mục 7 bước cuối) và Giai đoạn 3 đảm
  bảo có bản nộp hợp lệ độc lập với việc Giai đoạn 2 có hoàn thành đủ candidate hay không, luôn
  có 1 bản `submission.zip` hợp lệ sẵn sàng ở bất kỳ thời điểm nào trong quá trình — không rơi
  vào tình huống "đang giữa chừng, không có gì để nộp".

## 12. Tiêu chí hoàn thành giai đoạn này

- Cả 7 scene có ít nhất kết quả Giai đoạn 1 (5 biến thể đã so sánh, đã chọn winner).
- `bonsai` và `chair` có thêm kết quả Giai đoạn 2 (bounded search).
- `submission.zip` cuối cùng (Giai đoạn 3) qua được `validate_submission` không lỗi, QA mắt
  thường không phát hiện ảnh bất thường.
- Điểm trung bình 7 scene đo trên holdout được so sánh trực tiếp với baseline 0.622 hiện tại để
  xác nhận có cải thiện thật (không giả định đạt đúng 0.80 — đây là mục tiêu định hướng, không
  phải cam kết cứng, vì phụ thuộc dữ liệu thật chưa biết trước).
