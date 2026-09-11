# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- Record này mô tả toàn ảnh như thế nào?

  * Gán **duy nhất một nhãn phân loại** đại diện cho **toàn bộ bức ảnh** (Image-level classification), không chỉ định vị trí hay tọa độ/ranh giới của từng vật thể cụ thể trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?

  - Do tập dữ liệu huấn luyện chuẩn ( **ImageNet-1K** ) và tác giả/nhóm phát triển mô hình (Ultralytics) định nghĩa sẵn từ trước.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

  * `class_id`: Giúp máy tính xử lý và lưu trữ dữ liệu chính xác, tránh nhầm lẫn do chuỗi ký tự.
  * `class_name`: Giúp người đọc/reviewer dễ hiểu tên đối tượng bằng ngôn ngữ tự nhiên.
  * `taxonomy_name` (ví dụ `ImageNet-1K` vs `COCO-80`): Định danh không gian nhãn/chuẩn phân loại đang dùng, tránh xung đột khi tích hợp nhiều mô hình hoặc tập dữ liệu khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

  * Guideline cần quy định rõ tiêu chí chọn nhãn: ưu tiên chủ thể nằm ở trung tâm/nổi bật nhất, chủ thể chiếm diện tích lớn nhất, hoặc chuyển sang bài toán **multi-label classification** (gán nhiều nhãn cho một ảnh).
- Vì sao model score không phải ground truth?

  - Model score (độ tin cậy) chỉ là xác suất toán học do mô hình tự tính dựa trên trọng số đã học. Ground truth phải là sự thật khách quan do con người (annotator/reviewer) xác nhận.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- Diễn giải vị trí box bằng lời:
  * Vật thể thuộc lớp `"oven"` được bao bởi hộp chữ nhật có góc trên-trái tại tọa độ $(x=12.5, y=140.2)$ pixel và góc dưới-phải tại $(x=310.8, y=420.1)$ pixel (tính từ gốc góc trên-trái của ảnh), có chiều rộng $298.3$ pixel và chiều cao $279.9$ pixel.
- So sánh số prediction ở hai threshold:
  * Tại `threshold = 0.20`: Số lượng prediction nhiều hơn (giữ lại cả những dự đoán mờ/nhiễu).
  * Tại `threshold = 0.60`: Số lượng prediction ít hơn (chỉ giữ lại các dự đoán có độ tin cậy cao).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  * Threshold thấp tăng độ bao phủ (ít bỏ sót vật thể - recall cao) nhưng tạo ra nhiều báo động giả (false positives), khiến reviewer tốn nhiều công sức để xóa bớt.
  * Threshold cao làm giảm báo động giả nhưng dễ bỏ sót vật thể thật (false negatives).
- Đề xuất một quy tắc box chặt:
  * Bounding box phải bao sát viền ngoài cùng của đối tượng (tight bounding box), khoảng trống thừa không vượt quá $5%$ diện tích box và không cắt lẹm vào đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  * Guideline cần nêu rõ: Nếu đối tượng bị che khuất $>70%$, có vẽ box không? Với đối tượng cắt mép ảnh, vẽ box chỉ phần nhìn thấy hay bao gồm cả phần suy đoán ngoài khung hình?

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?

  * Polygon mô tả chính xác ranh giới/hình dạng thực tế (boundary) theo pixel của đối tượng, giúp loại bỏ hoàn toàn phần nền xung quanh nằm bên trong bounding box.
- `instance_id` dùng để làm gì và không phải loại ID nào?

  * `instance_id` dùng để phân biệt và định danh từng cá thể đối tượng riêng lẻ trong ảnh (kể cả khi chúng cùng lớp). Nó **không phải** là `class_id` (mã lớp chung) và **không phải** là tracking ID qua nhiều frame video.
- Đề xuất một quy tắc biên mask:

  * Đường biên đa giác (polygon) phải bám sát ranh giới viền ngoài của đối tượng (pixel-level accuracy), không ăn sâu vào thân đối tượng và không chờm sang vùng nền/vật thể khác.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

  - Cần quy định: Khi 2 đối tượng đè lên nhau thì mask của đối tượng nằm trước có được đè lên đối tượng nằm sau không, và vùng bóng đổ (shadow)/vùng mờ chuyển động (motion blur) có được tính vào mask không.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                        | Lỗi hoặc điểm mơ hồ quan sát được                              | Annotator làm gì?                                                              | Reviewer xem gì?                                                                                    |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | Class ID/Tên lớp duy nhất                               | Ảnh có nhiều vật thể ngang hàng nhau                               | Chọn nhãn của vật thể chính/nổi bật nhất theo ưu tiên trong guideline | Kiểm tra nhãn được chọn có đại diện đúng chủ thể chính của ảnh không               |
| Phát hiện vật thể | Bounding box$[x_1,y_1,x_2,y_2]ư$ + Class ID             | Box vẽ quá lỏng (nhiều nền) hoặc bỏ sót vật thể bị che khuất | Vẽ box bao sát viền vật thể, gán đúng lớp                               | Kiểm tra độ chặt của box, sự trùng lặp và vật thể bị bỏ sót                            |
| Instance segmentation | Đa giác (Polygon set of points) + Instance ID + Class ID | Đường biên đa giác bị răng cưa, dính sang vật thể bên cạnh | Vẽ đa giác bám chính xác viền điểm ảnh của từng instance             | Kiểm tra độ mượt/chính xác của đường biên và việc tách biệt các instance cùng lớp |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  * Tuyệt đối không tải lên hoặc lưu trữ thông tin định danh cá nhân (PII như họ tên, MSSV, SĐT, khuôn mặt nhận diện cá nhân), dữ liệu riêng tư hoặc dữ liệu chưa được cấp phép trong dự án / repository / Colab.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  * Lecturer / LabCoach

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
