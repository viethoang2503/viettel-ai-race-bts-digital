# Thiết kế: Pipeline Novel View Synthesis cho VAR 2026 - Digital Twin trạm BTS

## 1. Bối cảnh & mục tiêu

Cuộc thi VAR 2026 - Digital Twin cho trạm BTS, vòng 1, yêu cầu tái dựng cấu trúc 3D ngầm định
từ ảnh drone/hand-held đa góc nhìn và sinh ảnh RGB tại các pose camera chưa từng xuất hiện
trong tập train. Dữ liệu gồm 7 scene (`VAI_NVS_DATA_ROUND2/`): 5 scene BTS thật (HCM0421,
HCM0539, HCM0540, HCM0644, HCM0674 - ảnh drone DJI, 240 ảnh train / 60 pose test mỗi scene,
1320x989) và 2 scene tham chiếu/benchmark (`chair`, `bonsai` - `bonsai` là scene chuẩn
Mip-NeRF360 công khai, dùng để đối chiếu số liệu tự đo với số liệu published).

**Ràng buộc chính:**
- Deadline: 30/07/2026 (viết spec ngày 18/07/2026, còn ~12 ngày).
- Hạ tầng: Google Colab Pro/Pro+ để train/render (GPU); máy local không có GPU, dùng để code,
  quản lý git, đóng gói submission.
- Hạ tầng inference chính thức của BTC: 1x RTX A4000 (20GB VRAM), 4-8 CPU cores, 16-32GB RAM -
  checkpoint cuối cùng phải chạy vừa trong giới hạn này dù train trên GPU mạnh hơn.
- Không có ground-truth cho `test_poses.csv` thật (private test) - toàn bộ đánh giá trong quá
  trình phát triển phải dựa trên holdout tự tạo từ tập train.
- Mục tiêu: **tối đa hoá điểm số cuối cùng** theo công thức
  `Score = 0.4x(1-LPIPS) + 0.3xSSIM + 0.3xPSNR_norm`, không phải tối đa hoá số lượng kỹ thuật
  sử dụng. Thiết kế phải có cơ chế tự chọn cấu hình tốt nhất theo dữ liệu thật, không giả định
  trước.
- Tuân thủ mục 10 của đề bài: không dùng dữ liệu ngoài, không truy xuất/suy đoán ground-truth
  test, phải tái lập được kết quả (config, log, checkpoint), không chỉnh sửa thủ công ảnh đầu ra.

**Lựa chọn engine:** bám theo repo gốc `graphdeco-inria/gaussian-splatting` (quyết định của
người dùng, ưu tiên kiểm soát code toàn bộ hơn là dùng thư viện tích hợp sẵn như nerfstudio/gsplat).

## 2. Kiến trúc tổng thể & luồng làm việc

```
[Local: viết code, git]  --push-->  [GitHub repo]  --pull-->  [Colab: GPU compute]
                                                                      |
                                                              mount Google Drive
                                                                      |
                                                    [Drive: dataset, checkpoint,
                                                     CUDA cache, logs, kết quả render]
                                                                      |
                                                          <--download kết quả--
[Local: evaluate, đóng gói submission.zip]
```

- **Local (không GPU):** source of truth cho code (git repo). Chạy các bước không cần GPU:
  validate data, tính metric trên ảnh đã tải về, đóng gói/validate submission cuối cùng.
- **GitHub:** cầu nối code giữa local và Colab; đồng thời phục vụ yêu cầu tái lập kết quả
  (mục 10.3).
- **Google Drive:** lưu trữ bền giữa các session Colab (session có thể bị ngắt bất kỳ lúc nào
  dù dùng Pro/Pro+) - chứa dataset gốc, CUDA extension đã compile sẵn, checkpoint theo từng
  scene/biến thể, training logs, ảnh render.
- **Colab:** chỉ chạy phần cần GPU (train, render). Mọi bước train phải resume-safe.

## 3. Cấu trúc repo

```
var2026-nvs-bts/
├── .gitignore                  # loại trừ dataset, checkpoint, output lớn
├── environment/
│   ├── setup_colab.sh          # cài đặt, ưu tiên restore CUDA ext từ Drive cache
│   └── requirements.txt
├── src/
│   ├── data_validation/
│   │   └── validate_scene.py   # đối chiếu images.bin vs thư mục, parse test_poses.csv,
│   │                            # tính bounding box scene, kiểm tra camera model
│   ├── training/
│   │   ├── configs/            # yaml hyperparam mặc định + override theo scene/biến thể
│   │   ├── train_wrapper.py    # wrap train.py gốc: auto-resume, hooks cho depth-reg,
│   │   │                        # appearance embedding, VRAM budget guard
│   │   └── patches/            # patch rasterizer (anti-aliasing kiểu Mip-Splatting)
│   ├── rendering/
│   │   └── render_from_csv.py  # đọc test_poses.csv -> dựng Camera theo từng dòng -> render
│   ├── postprocess/
│   │   └── prune_floaters.py   # cleanup Gaussian sau train (opacity, bounding box, scale)
│   ├── evaluation/
│   │   ├── make_holdout_split.py   # holdout theo rìa quỹ đạo, không random đều
│   │   ├── compute_metrics.py      # LPIPS/SSIM/PSNR đúng công thức mục 8.4
│   │   └── select_best_config.py   # tự chọn biến thể tốt nhất theo scene dựa trên holdout
│   ├── orchestrator/
│   │   └── run_all.py          # loop scene x biến thể: train -> render -> eval -> tổng hợp
│   └── submission/
│       ├── package_submission.py
│       └── validate_submission.py  # kiểm tra đủ scene, đúng tên/kích thước trước khi zip
├── notebooks/
│   └── colab_runner.ipynb      # entrypoint chạy trên Colab: mount Drive, setup env, gọi orchestrator
├── configs/
│   └── scenes.yaml             # danh sách 7 scene + metadata (resolution, camera model...)
├── tests/
│   ├── test_pose_conversion.py
│   └── test_validate_submission.py
└── docs/superpowers/specs/     # design docs (tài liệu này)
```

## 4. Môi trường & quản lý session Colab

- `setup_colab.sh`: mount Drive, kiểm tra CUDA extension (`diff-gaussian-rasterization`,
  `simple-knn`, và bản patch anti-aliasing) đã build sẵn trên Drive chưa; nếu có, copy thẳng
  vào site-packages (bỏ qua compile, tiết kiệm 5-10 phút/session); nếu chưa, compile rồi lưu
  lại lên Drive cho các session sau.
- Mọi script train nhận flag resume: kiểm tra checkpoint mới nhất trên Drive cho
  scene+biến thể đang chạy, tiếp tục thay vì train lại từ đầu.
- Log training (tensorboard + file JSON tóm tắt loss/iteration) ghi trực tiếp ra Drive theo
  thời gian thực, không giữ trong runtime tạm thời của Colab.

## 5. Data validation module

- Đối chiếu số ảnh registered trong `images.bin` với số file thực tế trong `train/images/`.
- Parse toàn bộ `test_poses.csv`, validate schema và kiểu dữ liệu từng cột.
- Xác nhận camera model (PINHOLE/SIMPLE_PINHOLE) và số lượng camera entry trong `cameras.bin`.
- Tính bounding box của scene từ `points3D.bin` (dùng cho floater pruning ở mục 8).
- Chạy 1 lần cho cả 7 scene trước khi bắt đầu train; output báo cáo cảnh báo nếu bất thường.

## 6. Training pipeline: ma trận thực nghiệm theo biến thể kỹ thuật

Thay vì giả định trước "kỹ thuật nào cũng tốt", mỗi scene được train với **nhiều biến thể**,
đánh giá trên holdout, rồi tự động chọn biến thể tốt nhất (mục 9). Các biến thể:

| Biến thể | Mô tả | Rủi ro tích hợp |
|---|---|---|
| `baseline` | 3DGS gốc, không sửa đổi | Thấp - mốc tham chiếu bắt buộc |
| `+floater_cleanup` | baseline + post-process prune (opacity thấp, ngoài bounding box, scale bất thường) | Thấp - thuần Python, không đụng CUDA |
| `+depth_reg` | baseline + loss L1 giữa depth render và sparse depth chiếu từ `points3D.bin` (chỉ dùng dữ liệu đã cấp, không dùng model depth ngoài để tránh rủi ro tuân thủ) | Trung bình - cần sửa render để xuất depth map + sửa training loop |
| `+anti_alias` | thay rasterizer bằng bản patch kiểu Mip-Splatting (3D smoothing filter + 2D screen-space low-pass filter) | Cao - đụng code CUDA/C++, cần merge patch cẩn thận |
| `+appearance_embed` | mỗi ảnh train có affine color embedding học được, bù exposure; khi render test pose dùng embedding trung bình (mean) của toàn bộ ảnh train làm "appearance chuẩn" | Trung bình - thêm tham số học được + sửa loss |
| `full_stack` | kết hợp tất cả các kỹ thuật trên | Cao nhất - rủi ro tương tác giữa các kỹ thuật |

Thứ tự triển khai: theo độ rủi ro tăng dần (baseline -> floater_cleanup -> depth_reg ->
anti_alias -> appearance_embed -> full_stack), nhưng **tất cả đều được triển khai** theo yêu
cầu "hoàn chỉnh nhất có thể".

## 7. Giới hạn VRAM theo hạ tầng inference thật (A4000 20GB)

Colab Pro/Pro+ có thể cấp GPU mạnh hơn A4000 (V100/A100). Nếu checkpoint có số lượng Gaussian
quá lớn, khi BTC chạy inference thật trên A4000 20GB có thể OOM và fail toàn bộ scene đó -
rủi ro nghiêm trọng hơn điểm số thấp.

- Sau khi train xong mỗi biến thể, đo VRAM cần thiết để render checkpoint đó (giả lập giới hạn
  20GB bằng `torch.cuda.set_per_process_memory_fraction` hoặc theo dõi peak memory khi render).
- Nếu vượt ngưỡng an toàn (đặt margin, vd 16GB để chừa buffer), checkpoint đó bị loại khỏi danh
  sách ứng viên ở bước chọn cấu hình tốt nhất (mục 9), dù điểm holdout có cao.
- Đưa giới hạn số Gaussian tối đa (qua `--densify_grad_threshold`, `--densify_until_iter`) vào
  config để chủ động kiểm soát thay vì phát hiện muộn sau khi train xong.

## 8. Render-from-CSV module

- Đọc từng dòng `test_poses.csv`, convert quaternion COLMAP sang rotation matrix bằng đúng
  công thức baseline dùng (`qvec2rotmat` trong `scene/colmap_loader.py`).
- Dựng `Camera` object với FoV tính từ `fx, fy, width, height` **riêng theo từng dòng** (không
  dùng resolution cố định của tập train).
- Có unit test riêng cho phần convert pose (mục 12) - đây là chỗ dễ lỗi âm thầm nhất (sai
  convention quaternion cho ra ảnh sai góc nhìn nhưng không crash).

## 9. Post-processing: floater cleanup & auto-select best config

- `prune_floaters.py`: loại Gaussian có opacity thấp, Gaussian ngoài bounding box (từ mục 5) +
  margin, Gaussian có scale bất thường lớn.
- `select_best_config.py`: với mỗi scene, so sánh điểm holdout (mục 10) của tất cả biến thể đã
  train (đã qua kiểm tra VRAM ở mục 7), chọn biến thể có `Score` holdout cao nhất làm cấu hình
  chính thức để render `test_poses.csv` thật. Ghi lại lý do chọn (bảng điểm so sánh) vào log
  tái lập kết quả.

## 10. Evaluation framework (không cần ground-truth thật)

- `make_holdout_split.py`: tách ảnh train theo pose ở "rìa" vùng phủ quỹ đạo bay (không random
  đều) để mô phỏng đúng độ khó extrapolation của test thật - vì drone bay quỹ đạo orbit/spiral
  và test pose nhiều khả năng nằm ngoài phạm vi quỹ đạo train dày đặc.
- `compute_metrics.py`: implement đúng công thức mục 8.4, tự chọn và cố định `PSNR_max` hợp lý,
  log kết quả theo từng scene + từng biến thể để so sánh trong `select_best_config.py`.
- Chạy validate trên `bonsai` trước tiên, đối chiếu với số liệu published của Mip-NeRF360
  benchmark để xác nhận hàm tính metric đúng trước khi tin vào số đo trên 5 scene BTS.

## 11. Orchestrator

- `run_all.py`: loop qua 7 scene x 6 biến thể (mục 6) -> train (resume-safe) -> render holdout
  -> tính metric -> kiểm tra VRAM -> ghi kết quả tổng hợp vào một bảng duy nhất
  (`results_summary.csv`). Đây là entrypoint chính chạy trong `colab_runner.ipynb`.

## 12. Testing/QA

- Unit test pose conversion (mục 8) - so khớp với công thức COLMAP tham chiếu bằng vài case
  đã biết trước kết quả.
- Integration test full pipeline trên `bonsai` (scene nhỏ, có benchmark tham chiếu).
- Test `validate_submission.py` với input lỗi cố ý (thiếu ảnh, sai tên, sai kích thước) để đảm
  bảo nó bắt được lỗi trước khi nộp thật.

## 13. Visual QA trước khi đóng gói

Metric tự động trên holdout không chắc phản ánh đúng chất lượng trên test pose extrapolate
thật. Trước khi zip nộp: xem thủ công một mẫu ảnh render từ `test_poses.csv` thật (mỗi scene
vài ảnh) để phát hiện lỗi rõ ràng (ảnh đen, floater lớn, sai hoàn toàn góc nhìn). Đây là bước
**kiểm tra chất lượng**, không phải "chỉnh sửa thủ công ảnh đầu ra" (vẫn cấm theo mục 10.4) -
không sửa ảnh, chỉ dùng để quyết định có cần train lại/điều chỉnh config hay không trước
deadline.

## 14. Submission packaging & validation

- `package_submission.py`: gom ảnh render theo cấu trúc `<submission_dir>/<image_name>` (đúng
  tên file và extension gốc từ `test_poses.csv`, không đổi thành `.png`) đúng yêu cầu mục 7 của
  đề bài. `<submission_dir>` là giá trị config theo từng scene (mặc định bằng tên scene thật,
  vd `HCM0421`, `chair`) - ví dụ `scene_001/0001.png` trong đề bài chỉ là minh hoạ cấu trúc,
  **cần xác nhận lại với BTC** xem có bắt buộc đặt tên `scene_001` kiểu số thứ tự hay không,
  vì không thể suy ra chắc chắn từ dữ liệu được cấp.
- `validate_submission.py`: bắt buộc chạy trước khi nộp - kiểm tra đủ số scene, đúng tên file
  theo `image_name` trong `test_poses.csv`, đúng `width x height` từng ảnh, và **không có file
  hoặc scene thừa** (kể cả rác như `__MACOSX/`) so với danh sách mong đợi. Thiếu hoặc thừa 1
  scene/file sẽ làm mất điểm toàn bộ theo quy định mục 8.4, đây là rủi ro vận hành cần phòng
  tuyệt đối.

## 15. Reproducibility bundle

Mỗi scene sau khi chọn cấu hình cuối cùng (mục 9): lưu config yaml, training log, checkpoint
vào Drive theo cấu trúc cố định, kèm bảng so sánh điểm các biến thể đã thử - sẵn sàng cung cấp
nếu BTC yêu cầu theo mục 10.3. Ghi rõ nguồn gốc patch anti-aliasing (tham khảo từ Mip-Splatting,
cùng dạng giấy phép non-commercial nghiên cứu như 3DGS gốc, phù hợp bối cảnh thi học thuật).

## 16. Rủi ro & giới hạn còn tồn tại

- Test pose thật có thể extrapolate mạnh hơn holdout tự tạo dự đoán - không có cách loại bỏ
  hoàn toàn rủi ro này khi không có ground-truth thật.
- Tích hợp anti-aliasing patch (CUDA) là phần rủi ro kỹ thuật cao nhất về thời gian - nếu phát
  sinh lỗi khó sửa gần deadline, biến thể `full_stack`/`+anti_alias` có thể bị loại tự động bởi
  `select_best_config.py` (vì không train xong hoặc lỗi khi render), pipeline vẫn tự rơi về
  biến thể tốt nhất trong số các biến thể đã train thành công - đây chính là lưới an toàn của
  thiết kế ma trận thực nghiệm (mục 6, 9).
- Chưa xác nhận `PSNR_max` chính xác BTC dùng - tự chọn giá trị hợp lý dựa trên PSNR điển hình
  của NVS thật, chấp nhận sai số nhỏ trong ước lượng điểm holdout.
