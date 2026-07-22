# VAR 2026 — Digital Twin trạm BTS: Đề bài & Phân tích tổng hợp

## PHẦN 1: ĐỀ BÀI GỐC

### 1.1 Mô tả bài toán

Xây dựng hệ thống AI có khả năng tái dựng cấu trúc 3D ngầm định của trạm BTS từ tập ảnh drone, và sinh ảnh RGB tại các góc nhìn chưa từng được chụp. Đây là bài toán **Novel View Synthesis (NVS)** phục vụ xây dựng Digital Twin — bản sao số 3D độ chính xác cao của hạ tầng viễn thông cho mục đích giám sát, kiểm tra, bảo trì và quy hoạch lắp đặt thiết bị.

- Mỗi scene: 100-300 ảnh RGB kèm thông số camera + pose tương ứng.
- Cần sinh ảnh tại 20-50 góc nhìn mục tiêu, đảm bảo đúng hình học, đúng vị trí thiết bị, chất lượng hình ảnh chân thực.
- Dữ liệu thu thập từ drone bay quanh đối tượng hoặc camera cầm tay.
- Đối tượng: trạm BTS, công trình hạ tầng, các đối tượng thực tế khác.
- Lĩnh vực: Computer Vision, 3D Vision, Neural Rendering, Novel View Synthesis, Digital Twin.

### 1.2 Cấu trúc dữ liệu

```
├── train/
│   ├── images/          : Ảnh training
│   ├── sparse/0/         : Sparse reconstruction từ COLMAP
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
└── test/
    └── test_poses.csv   : Camera poses cho test images
```

- Train images: ~80%, Test images: ~20% (poses).
- Camera poses và sparse reconstruction đã được dựng sẵn bằng COLMAP, cung cấp cho thí sinh.

### 1.3 Format `test_poses.csv`

```
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```
- `image_name`: tên ảnh đầu ra cần sinh
- `qw,qx,qy,qz`: quaternion rotation theo convention COLMAP
- `tx,ty,tz`: camera translation
- `fx,fy`: focal length; `cx,cy`: principal point
- `width,height`: kích thước ảnh cần sinh

### 1.4 Đầu vào / Đầu ra

**Đầu vào:** tập ảnh train đa góc nhìn, camera intrinsics, camera poses, sparse reconstruction COLMAP, danh sách test poses.

**Đầu ra:** ảnh RGB tương ứng toàn bộ test poses, đúng cấu trúc hình học, đúng vị trí vật thể, chất lượng chân thực và nhất quán.

### 1.5 Format submission

```
submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   └── ...
```
Yêu cầu: đúng số lượng và tên scene, đúng tên file, đúng kích thước ảnh, đúng số lượng ảnh mỗi scene.

### 1.6 Metrics đánh giá

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| LPIPS | Tương đồng cảm quan (deep feature) | Càng thấp càng tốt |
| SSIM | Tương đồng cấu trúc ảnh | Càng cao càng tốt |
| PSNR | Sai số mức pixel | Càng cao càng tốt |

```
psnr_norm = clamp(psnr_val / psnr_max, 0.0, 1.0)
Score = 0.4 × (1 - LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

Điểm bảng xếp hạng = điểm trung bình toàn bộ scene. **Thiếu hoặc thừa scene so với ground-truth → kết quả không được tính** (rủi ro nghiêm trọng nhất về vận hành).

### 1.7 Quy định chống gian lận

- **Cấm dữ liệu ngoài:** không dùng ảnh/video/3D data ngoài chứa cùng đối tượng/scene; không thu thập thêm dữ liệu thực địa hay từ Internet liên quan trực tiếp scene thi; không dùng bất kỳ nguồn nào để tái tạo/suy luận ground-truth test.
- **Cấm truy xuất/suy đoán dữ liệu kiểm thử:** không truy cập trái phép ground-truth, không khai thác lỗ hổng hệ thống.
- **Yêu cầu khả năng tái lập:** đội thứ hạng cao phải cung cấp mã nguồn train/inference, config, danh sách thư viện, checkpoint, training log — chứng minh tái lập được từ pipeline đã công bố.
- **Cấm chỉnh sửa thủ công ảnh đầu ra:** toàn bộ ảnh phải được sinh tự động bởi thuật toán/mô hình AI; cấm chỉnh sửa bằng phần mềm đồ họa, ghép ảnh, can thiệp thủ công vào từng test pose.

### 1.8 Baseline tham khảo
`https://github.com/graphdeco-inria/gaussian-splatting`

### 1.9 Thông tin riêng Vòng 1

| Hạng mục | Thông tin |
|---|---|
| Số ảnh/scene | 150-300 ảnh RGB |
| Số pose mục tiêu/scene | 40-70 |
| Dung lượng | 200-300 MB |
| Công bố private test #1 | 02/07/2026 |
| Deadline submission | 30/07/2026 |
| Hạ tầng inference tham khảo | 1× RTX A4000 (20GB VRAM), 4-8 CPU cores, 16-32GB RAM |

Thí sinh có thể submit nhiều lần trong thời gian mở; hệ thống ghi nhận **bản submit cuối cùng** trước deadline. Ban tổ chức khuyến nghị kiểm tra kỹ pipeline trên public set trước khi chạy trên private test, và ước lượng thời gian chạy để đảm bảo kịp deadline.

---

## PHẦN 2: PHÂN TÍCH ĐỀ BÀI

### 2.1 Đối chiếu với dữ liệu thực tế đã kiểm tra

Đã đọc trực tiếp `VAI_NVS_DATA_ROUND2/` — 7 scene tổng dung lượng 1.6GB:

| Scene | Train | Test poses | Resolution | Loại |
|---|---|---|---|---|
| HCM0421, HCM0539, HCM0540, HCM0644, HCM0674 | 240 | 60 | 1320×989 | Ảnh drone DJI thật — trọng tâm chấm điểm |
| chair | 205 | 58 | 720×1280 | Video hand-held |
| bonsai | 248 | 28 | 1920×1080 | Scene chuẩn Mip-NeRF360 công khai |

**Đính chính (2026-07-22):** cả 7 scene đều nằm trong gói private test #1 BTC cấp và **đều cần có mặt trong `submission.zip`** — xác nhận trực tiếp từ người dùng, phụ trách bài thi. Nhận định trước đó trong tài liệu này (cho rằng `chair`/`bonsai` chỉ là scene tham chiếu, không cần nộp) là **sai**, đã dẫn đến bỏ sót 2 scene khi đóng gói submission đầu tiên — xem `notebooks/colab_runner_bonsai.ipynb` (đổi tên/mục đích: giờ dùng để train `bonsai`+`chair` cho submission thật, không chỉ để validate) và bước ghép 7 scene ở `notebooks/colab_runner_hcm.ipynb` Bước 8.

Xác nhận: `test/` chỉ chứa `test_poses.csv`, **không có ảnh ground-truth** — đúng với quy định cấm truy xuất test data. `cameras.bin` có `cx=width/2, cy=height/2` chính xác ở cả 7 scene → camera model PINHOLE/SIMPLE_PINHOLE, tương thích thẳng với baseline. Sparse point cloud dày (15-22MB/scene ở nhóm HCM) → khởi tạo Gaussian tốt.

### 2.2 So sánh phương pháp kỹ thuật

- **3DGS (baseline):** rasterization, train nhanh, tận dụng tốt sparse reconstruction có sẵn — lựa chọn hợp lý của BTC.
- **NeRF/Instant-NGP:** hình học mượt hơn, ít floater rời rạc, nhưng chậm hơn và khó đạt chi tiết cao ở cấu trúc mảnh (ăng-ten, dây cáp).
- **Biến thể nâng cấp từ 3DGS:** Mip-Splatting (anti-aliasing multi-scale — hợp với ảnh drone có cả cận cảnh/viễn cảnh), 2D Gaussian Splatting (hình học chính xác hơn), depth/normal regularization (giảm floater), appearance embedding (bù lệch exposure).

### 2.3 Thách thức đặc thù dữ liệu drone/BTS

- **Extrapolation risk:** drone bay quỹ đạo orbit/spiral độ cao gần cố định; test pose (20-50 góc) nhiều khả năng nằm ngoài phạm vi quỹ đạo train dày đặc — khác hẳn benchmark NeRF tiêu chuẩn.
- **Multi-scale:** cận cảnh trạm + viễn cảnh nền/đất dễ sinh floater — baseline 3DGS tự cảnh báo vấn đề này.
- **Sky/nền vô cực:** ảnh chụp từ cao có mảng trời lớn không parallax → floater lơ lửng, gây hại nặng cho LPIPS.
- **Exposure lệch** giữa các ảnh do auto-exposure trong lúc bay.
- **Cấu trúc mảnh** (ăng-ten, dây cáp): khó cho cả COLMAP matching lẫn Gaussian bám sát.
- Ảnh đã downsample 1/4 theo README → giảm nhiễu/motion-blur, là điểm thuận lợi.

### 2.4 Chiến lược tối ưu điểm số

- LPIPS chiếm 40% trọng số, phạt nặng **blur và floater** hơn là sai lệch pixel nhỏ → ưu tiên cleanup floater/tăng độ nét mang lại ROI cao hơn ép PSNR.
- PSNR_norm bị clamp theo `PSNR_max` tự chọn của BTC (không biết trước) → có điểm bão hòa, không nên cố vắt thêm PSNR biên tế.
- SSIM tương quan với đúng cấu trúc hình học — cải tiến hình học lợi kép cho cả SSIM và LPIPS.
- **Rủi ro lớn nhất không phải thuật toán mà là tính đầy đủ submission** — thiếu 1 scene mất điểm toàn bộ → bắt buộc có script tự validate trước khi nộp.

### 2.5 Cách xử lý việc không có ground-truth

- Giai đoạn phát triển: tự tạo holdout từ ảnh train (chọn ở rìa vùng phủ quỹ đạo, không random đều, để mô phỏng đúng độ khó extrapolation) — dùng cờ `--eval` sẵn có trong baseline.
- Giai đoạn nộp bài: train lại 100% dữ liệu train, render "mù" theo `test_poses.csv` thật (không đối chiếu được), dựa vào QA thủ công bằng mắt + độ tin cậy của bước validate ở trên.
- `bonsai` cũng dùng để kiểm chứng hàm tính metric của mình có khớp với số liệu Mip-NeRF360 published hay không, trước khi tin vào số đo trên 5 scene BTS — **nhưng đây là lợi ích PHỤ**, không phải lý do duy nhất nó có trong dataset: cả `bonsai` và `chair` đều phải xuất hiện trong `submission.zip` cuối cùng (xem đính chính ở mục 2.1).

### 2.6 Định hướng triển khai đã thống nhất

- Bám theo baseline gốc `graphdeco-inria/gaussian-splatting` (không dùng nerfstudio/gsplat).
- Hạ tầng: train/render trên Google Colab Pro/Pro+, code + git repo quản lý ở local, đồng bộ qua GitHub.
- Mục tiêu: tối đa hoá điểm số thật — pipeline train nhiều biến thể kỹ thuật (floater cleanup, depth reg từ sparse COLMAP, anti-aliasing kiểu Mip-Splatting, appearance embedding) cho mỗi scene, tự động chọn biến thể tốt nhất theo holdout, có guard VRAM cho hạ tầng inference A4000 20GB.
- Deadline 30/07/2026 — còn khoảng 12 ngày kể từ khi bắt đầu phân tích (18/07/2026).

---

## Tài liệu liên quan trong repo

- Spec thiết kế: `docs/superpowers/specs/2026-07-18-nvs-bts-pipeline-design.md`
- Plan triển khai (baseline core pipeline): `docs/superpowers/plans/2026-07-18-core-nvs-pipeline.md`
