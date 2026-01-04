# Casual Discovery Transformer

Dự án nghiên cứu và triển khai mô hình Causal Discovery Transformer để dự đoán giá cổ phiếu và phát hiện mối quan hệ nhân quả giữa các biến tài chính.

## Mô tả

Project này bao gồm các mô hình Transformer được thiết kế để:
- Phát hiện cấu trúc nhân quả (causal structure) giữa các biến tài chính
- Dự đoán giá cổ phiếu dựa trên các chỉ báo kỹ thuật
- So sánh hiệu suất giữa Vanilla Transformer và Causal Discovery Transformer

## Cấu trúc Project

- `CD_Transformer.ipynb`: Mô hình Causal Discovery Transformer chính với sliding windows
- `CD_Transfomer_MODIFY_WITH_ATTENTION_GUIDED_CAUSUAL..ipynb`: Phiên bản cải tiến với attention-guided causal discovery
- `vanilla_transformer.ipynb`: Mô hình Vanilla Transformer để so sánh baseline
- `Casual_discovery_transformer.docx`: Tài liệu mô tả chi tiết về mô hình

## Tính năng chính

### Causal Discovery Transformer
- **CausalDiscoveryLayer**: Học cấu trúc đồ thị nhân quả giữa các biến
- **CausalAttention**: Self-attention với mask nhân quả thời gian
- **Sliding Windows Training**: Huấn luyện trên nhiều cửa sổ thời gian
- **Sparsity Scheduling**: Điều chỉnh độ thưa của đồ thị nhân quả
- **Intervention Robustness**: Kiểm tra tính ổn định của mô hình

### Vanilla Transformer
- Mô hình Transformer chuẩn với positional encoding
- Causal masking để tránh rò rỉ thông tin tương lai
- Baseline để so sánh hiệu suất

## Yêu cầu

```bash
torch
numpy
pandas
yfinance
scikit-learn
matplotlib
seaborn
```

## Cài đặt

```bash
pip install torch numpy pandas yfinance scikit-learn matplotlib seaborn
```

## Sử dụng

### 1. Causal Discovery Transformer

```python
from CD_Transformer import StockPriceCausalPredictor

# Khởi tạo mô hình
predictor = StockPriceCausalPredictor(
    feature_names=['Close', 'Volume', 'Returns', 'RSI', 'MACD', ...],
    d_model=64,
    num_heads=4,
    num_layers=2
)

# Huấn luyện
train_sliding_windows(X, y, feature_names, ...)

# Dự đoán
results = predictor.predict(X_test)
```

### 2. Vanilla Transformer

```python
from vanilla_transformer import VanillaTransformer

# Khởi tạo và huấn luyện
model = VanillaTransformer(num_features=6, d_model=64)
# ... training code ...
```

## Kết quả

Mô hình có khả năng:
- Phát hiện mối quan hệ nhân quả giữa các chỉ báo tài chính
- Dự đoán giá cổ phiếu với độ chính xác cao
- Ổn định qua các cửa sổ thời gian khác nhau

## Tác giả

[Tên của bạn]

## License

[Chọn license phù hợp: MIT, Apache 2.0, etc.]

## Trích dẫn

Nếu sử dụng code này trong nghiên cứu, vui lòng trích dẫn:

```
@misc{casual_discovery_transformer,
  author = {[Tên của bạn]},
  title = {Casual Discovery Transformer for Stock Price Prediction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/[username]/Casual-Discovery-Transformer}
}
```
