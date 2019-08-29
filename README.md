PersonCounting and Heatmap Generation via Surveillance camera

# Giới thiệu
Đây là toàn bộ chương trình bao gồm: code và hướng dẫn đầy đủ cho từng phần. Sử dụng phương pháp YOLOv3 để phát hiện vật thể, MGN và Deep Cosine Metric để rút trích đặc trưng, thuật toán theo vết đối tượng DeepSORT và cách để có được biểu đồ nhiệt.
# Yêu cầu
Ngôn ngữ Python: phiên bản 3.6.0 nếu phiên bản khác có thể phát sinh lỗi
Một vài yêu cầu khác về thư viện, framework:
* pip 19.0.3
* numpy
* sklearn
* OpenCV 3.4.0.12
* tensorflow 1.12.0
* keras 2.1.5
# Cài đặt
Link tải file model: [Model](https://drive.google.com/drive/folders/1N6rZx481kLOZ4d7Wa2EAZzmgLdhT9eRF?usp=sharing)
Link tải dataset và annotate tương ứng: 
Có 2 file mà bạn cần quan tâm và lưu ý
darknet_skipframe.py và evaluate.py
darknet_skipframe.py đây là file code chứa nội dung chính của đề tài, input sẽ là 1 đoạn video, output sẽ là video (chứa số lượng người và biểu đồ nhiệt được minh họa trực tiếp trên video).
evaluate.py sẽ là file chứa code để đánh giá hệ thống.
## Chạy chương trình chính
Theo dõi cách chạy chương trình sau đây:

**darknet_skipframe.py**
Cách nhập đầu vào cho chương trình:
python darknet_skipframe.py \
--model= 'model_data/mars-small128.pb'\
--max_cosine_distance = 0.9 \
--maxAge = 100 

Bên trong file darknet_skipframe.py đã chứa đẩy đủ document giải thích code.

**evalute.py**

# Tổng quan về các file quan trọng khác
detection.py: Detection base class.
kalman_filter.py: A Kalman filter implementation and concrete parametrization for image space filtering.
linear_assignment.py: This module contains code for min cost matching and the matching cascade.
iou_matching.py: This module contains the IOU matching metric.
nn_matching.py: A module for a nearest neighbor matching metric.
track.py: The track class contains single-target track data such as Kalman state, number of hits, misses, hit streak, associated feature vectors, etc.
tracker.py: This is the multi-target tracker class.
