# Thiết kế: Colab runner notebook + fix 3 lỗi review (cache CUDA, fail-closed submission, holdout rỗng)

## 1. Bối cảnh & mục tiêu

Pipeline baseline NVS (Task 1-13 của `docs/superpowers/plans/2026-07-18-core-nvs-pipeline.md`) đã
implement đầy đủ phần logic không cần GPU, với `run_baseline_pipeline` (`src/orchestrator/
run_pipeline.py`) nhận `train_fn`/`render_fn` dưới dạng inject để test được không cần CUDA. Phần
còn thiếu: (1) glue code THẬT gọi `train.py`/`gaussian_renderer.render()` của vendored
gaussian-splatting, và (2) notebook Colab (`notebooks/colab_runner.ipynb`, đã định sẵn ở spec gốc
section 3 nhưng chưa viết) để người dùng chỉ cần mở notebook trên Colab và chạy, không phải tự
gọi orchestrator thủ công.

Đồng thời, review phát hiện 3 lỗi cần fix trước khi notebook này chạy đúng trên dữ liệu thật:

- **High** (`environment/setup_colab.sh:29`): cache CUDA extension bằng `pip download --no-binary
  :all:` gần như chắc chắn hỏng ở lần chạy thứ 2, vì `diff-gaussian-rasterization`/`simple-knn`
  không đóng gói sdist chuẩn có `setup.py` round-trip được; `|| true` còn che mất lỗi download
  rỗng, khiến marker `.built` bị touch dù cache không có gì để restore.
- **Medium** (`src/orchestrator/run_pipeline.py:160`): sau khi package + validate submission zip,
  code luôn set `result.submission_zip = submission_zip` bất kể `validation_problems` có lỗi hay
  không — vi phạm chính triết lý fail-closed mà function này đã áp dụng cho `skipped_scenes`.
- **Low** (`src/evaluation/make_holdout_split.py:16`): `np.stack` crash với traceback khó hiểu nếu
  `camera_centers` rỗng, thay vì báo lỗi rõ ràng.

**Ràng buộc:**
- Máy local không có GPU/CUDA — toàn bộ code liên quan CUDA chỉ verify được thủ công trên Colab.
- Phạm vi vẫn baseline-only (biến thể kỹ thuật nâng cao là plan riêng, ngoài phạm vi này).
- Không phá vỡ các test hiện có (`tests/test_run_pipeline.py`, `tests/test_make_holdout_split.py`).

## 2. Kiến trúc tổng thể

```
[Colab notebook: notebooks/colab_runner.ipynb]
   |
   +-- Cell 1: git clone --recurse-submodules <origin> (idempotent)
   |
   +-- Cell 2: mount Drive, !bash environment/setup_colab.sh (cache fix)
   |
   +-- Cell 3 (Python): load_scenes(), load_lpips_model(),
   |         real_train_fn (src/training/gs_train_fn.py),
   |         real_render_fn (src/rendering/gs_render_fn.py),
   |         run_baseline_pipeline(..., output_root=<Drive path>)
   |
   +-- Cell 4: in per_scene_scores / skipped_scenes / validation_problems / submission_zip
```

`output_root` luôn trỏ vào một thư mục trên Drive (không phải `/content` tạm thời của Colab), vì
session Colab có thể bị ngắt bất kỳ lúc nào — mọi checkpoint/log/render phải sống sót qua việc
ngắt kết nối.

## 3. Fix 2 bug logic (Medium, Low)

### 3.1 `run_pipeline.py` — fail-closed sau khi validate submission

Sau `result.validation_problems = validate_submission(...)`: chỉ set `result.submission_zip =
submission_zip` khi `validation_problems` rỗng; ngược lại `None`. File zip vật lý vẫn giữ trên đĩa
(không xoá) để debug, chỉ không được coi là submission hợp lệ — nhất quán với cách
`skipped_scenes` đã xử lý phía trên trong cùng function.

### 3.2 `make_holdout_split.py` — guard input rỗng

Đầu `select_holdout_images`: nếu `camera_centers` rỗng, `raise ValueError("camera_centers is
empty, cannot select holdout images")` thay vì để `np.stack` crash với lỗi khó hiểu.

## 4. Fix cache CUDA extension (`environment/setup_colab.sh`)

Thay cơ chế `pip download --no-binary :all:` bằng copy artifact đã build:

- **Cache miss:** `pip install -q "$ext_src_dir"` như cũ, sau đó xác định các file đã cài vào
  `SITE_PACKAGES` cho package đó (`.so`, thư mục `dist-info`/`egg-link` tương ứng — tên package
  suy ra từ `ext_name` sau khi thay `-` bằng `_` theo convention pip), copy toàn bộ vào
  `$CUDA_EXT_CACHE/$ext_name/`. Chỉ `touch "$cache_marker"` SAU KHI copy thành công (bỏ `|| true`
  che lỗi).
- **Cache hit:** copy ngược từ `$CUDA_EXT_CACHE/$ext_name/` thẳng vào `SITE_PACKAGES`, bỏ qua
  hoàn toàn bước `pip install`/compile.
- Tập file chính xác cần copy (tên `.so`, có `egg-link` hay không tuỳ cách `setup.py` của 2
  submodule này đóng gói) **chưa thể xác nhận từ máy local** — đây là rủi ro đã biết, verify bằng
  checklist thủ công trên Colab thật (chạy 2 lần liên tiếp, disconnect/reconnect, xác nhận lần 2
  skip build và log CUDA vẫn `True`), cùng cách tiếp cận Task 13 gốc đã dùng.

## 5. Glue code GPU thật

### 5.1 `src/training/gs_train_fn.py::real_train_fn(scene, output_dir, iterations=30000) -> Path`

- Resume-safe: gọi `find_latest_checkpoint(output_dir)` trước; nếu checkpoint đã ở đúng
  `iterations` mục tiêu, trả về ngay, không train lại (quan trọng khi Colab bị ngắt giữa chừng rồi
  chạy lại notebook).
- Nếu chưa đủ: `build_train_argv(scene, output_dir, iterations, resume_checkpoint=<latest or
  None>, extra_args=["--checkpoint_iterations", str(iterations)])`, chạy qua
  `subprocess.run(argv, cwd="third_party/gaussian-splatting", check=True)`.
- Sau khi subprocess xong: `find_latest_checkpoint` lại; nếu không có hoặc chưa đủ iteration mục
  tiêu, raise lỗi rõ ràng thay vì âm thầm trả checkpoint dở dang cho bước render.
- **Test được ở local:** phần logic quyết định resume/skip và build argv, bằng cách mock
  `subprocess.run` và `find_latest_checkpoint` — không cần GPU cho phần này.

### 5.2 `src/rendering/gs_render_fn.py::real_render_fn(checkpoint, params_list, output_dir) -> list[Path]`

- Thêm `third_party/gaussian-splatting` vào `sys.path`, load checkpoint qua `torch.load` +
  `GaussianModel.restore(...)`.
- Với mỗi `CameraParams`, dựng `scene.cameras.Camera` (dùng ảnh placeholder trống đúng kích thước
  `width x height` làm input cho `PILtoTorch`, vì đây là test pose mới — không có ảnh gốc để load).
- Gọi `gaussian_renderer.render(camera, gaussians, pipe, background)`, convert tensor sang mảng
  `uint8` HWC, ghi ra qua `render_from_csv.render_all` (đã xử lý đúng tên file/extension theo
  `test_poses.csv`).
- Chữ ký chính xác của `GaussianModel.restore()` (có thể cần `OptimizationParams` giả dù không
  train) **chưa xác nhận được ở local** (không có CUDA) — flag là rủi ro verify-trên-Colab, cùng
  tinh thần với các mục "chưa xác nhận được, cần verify thực nghiệm" đã có sẵn trong plan gốc
  (Task 8b, Task 13).

## 6. `notebooks/colab_runner.ipynb`

4 cell theo đúng thứ tự ở mục 2. Cell 1 idempotent (skip clone nếu thư mục repo đã tồn tại, để
chạy lại notebook sau khi Colab disconnect không lỗi). Cell 3 in rõ từng scene: score, hay lý do
bị skip/validation fail, để người dùng biết ngay kết quả mà không cần đọc code.

## 7. Testing/QA

- `tests/test_run_pipeline.py`: thêm case validate_submission trả problems -> `submission_zip is
  None` dù zip vẫn được tạo trên đĩa.
- `tests/test_make_holdout_split.py`: thêm case input rỗng -> `ValueError`.
- `gs_train_fn.py`: unit test resume/skip logic + argv build bằng mock, không cần GPU.
- `gs_render_fn.py` và `setup_colab.sh`: không test tự động được (cần CUDA/Colab thật) — checklist
  verify thủ công, ghi rõ trong PR/commit thay vì giả định đúng.
- Notebook: không có test tự động; checklist thủ công (chạy full trên `bonsai` trước, scene nhỏ có
  benchmark tham chiếu, trước khi chạy cả 7 scene).

## 8. Rủi ro còn tồn tại

- Cách copy `.so`/`egg-link` để cache CUDA extension là suy luận hợp lý nhất từ cấu trúc 2
  submodule, nhưng **chỉ xác nhận được đúng/sai khi chạy thật trên Colab** — nếu sai, cần điều
  chỉnh lại đúng tập file cần copy sau lần chạy thật đầu tiên.
- `GaussianModel.restore()` cho mục đích render-only (không train tiếp) có thể cần tham số khác so
  với dùng trong `train.py` — cùng loại rủi ro, verify trên Colab thật.
