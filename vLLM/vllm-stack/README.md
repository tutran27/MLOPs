# vLLM Stack

Stack triển khai **vLLM** kết hợp **LiteLLM Proxy** và hệ thống **Observability** (Prometheus, Loki, Grafana Alloy, Grafana) bằng Docker Compose.

## Kiến trúc & Services

| Service | Image | Port | Chức năng |
| --- | --- | --- | --- |
| `vllm` | `vllm/vllm-openai:latest` | `8000` (internal) | Inference server phục vụ model local theo chuẩn OpenAI API |
| `litellm` | `ghcr.io/berriai/litellm:main-latest` | `4000` | AI Gateway / Proxy định tuyến request đến `vllm` và Groq API |
| `litellm_db` | `postgres:17` | `5432` (internal) | Database PostgreSQL lưu cấu hình, key và log usage cho LiteLLM |
| `prometheus` | `prom/prometheus:latest` | `9090` | Thu thập metrics từ `vllm` và `litellm` |
| `loki` | `grafana/loki:latest` | `3100` | Lưu trữ và truy vấn log tập trung |
| `alloy` | `grafana/alloy:latest` | - | Thu thập container logs của `vllm` và `litellm` đẩy về Loki |
| `grafana` | `grafana/grafana:latest` | `3000` | Dashboard trực quan hóa metrics (Prometheus) và logs (Loki) |

## Cấu trúc thư mục

```text
.
├── compose.yaml                              # Định nghĩa toàn bộ Docker Compose stack
├── .env                                      # Biến môi trường & secrets (không commit lên Git)
├── .gitignore
├── alloy/
│   └── config.alloy                          # Cấu hình Grafana Alloy thu thập Docker logs
├── grafana/
│   └── provisioning/
│       └── datasources/
│           ├── loki.yaml                     # Auto-provision datasource Loki
│           └── prometheus.yaml               # Auto-provision datasource Prometheus
├── litellm/
│   └── config.yaml                           # Cấu hình model routing, DB và Prometheus callback
├── loki/
│   └── loki.yaml                             # Cấu hình Loki (TSDB, retention 168h)
└── prometheus/
    └── prometheus.yml                        # Cấu hình scrape metrics vLLM & LiteLLM
```

## Yêu cầu hệ thống

- Docker & Docker Compose
- GPU NVIDIA + NVIDIA Container Toolkit (để chạy `vllm`)

## Cấu hình môi trường (`.env`)

Tạo file `.env` ở thư mục gốc của project (file này đã được thêm vào `.gitignore` để tránh lộ secrets):

```env
MODEL=RedHatAI/Qwen3.5-4B-quantized.w4a16
MODEL_SERVED=qwen3-4b
HF_TOKEN=hf_your_huggingface_token
GROQ_API_KEY=gsk_your_groq_api_key
GRAFANA_ADMIN_PASSWORD=your_grafana_password
LITELLM_UI_PASSWORD=your_litellm_ui_password
LITELLM_MASTER_KEY=sk-your_litellm_master_key
```

> **Lưu ý bảo mật:** Không commit file `.env` lên Git và nên thay đổi các mật khẩu mặc định (bao gồm mật khẩu PostgreSQL trong `compose.yaml` và `litellm/config.yaml`) trước khi triển khai production.

## Cách chạy

Khởi động toàn bộ stack:

```bash
docker compose up -d
```

Xem log của các service:

```bash
docker compose logs -f vllm litellm
```

Dừng stack:

```bash
docker compose down
```

## Truy cập & Kiểm thử

- **LiteLLM Proxy / UI:** `http://localhost:4000` (tài khoản UI: `admin` / `$LITELLM_UI_PASSWORD`)
- **Grafana:** `http://localhost:3000` (tài khoản: `admin` / `$GRAFANA_ADMIN_PASSWORD`)
- **Prometheus:** `http://localhost:9090`
- **Loki:** `http://localhost:3100`

Ví dụ gọi Chat Completions qua LiteLLM:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Xin chào!"}]
  }'
```

---

## Đánh giá hiệu năng (Performance Benchmark)

### 1. Môi trường & Phương pháp đo

| Thành phần | Thông số |
| --- | --- |
| **GPU** | NVIDIA GeForce RTX 3060 12GB (12,288 MiB VRAM, Compute Capability 8.6, Driver `595.99.02`) |
| **CPU / RAM** | Intel Core i5-13400F (16 threads) / 30 GiB RAM |
| **Model** | `Qwen/Qwen3-4B-AWQ` (AWQ 4-bit, `MarlinLinearKernel`, `FlashAttention v2`, trọng số `2.5 GiB`) |
| **Engine / Gateway** | `vLLM 0.34.0` (`http://vllm:8000`) + `LiteLLM Proxy` (`http://litellm:4000`, Postgres logging + Prometheus) |
| **Công cụ & Chuẩn đo** | `vllm bench serve` (`openai-chat`, streaming, `--ignore-eos`, `--num-warmups 3`, **seed khác nhau mỗi lần chạy** để loại bỏ sai lệch Prefix Cache) |
| **Kết quả thành công** | 100% request thành công trên toàn bộ kịch bản, `0` fallback ngoài ý muốn sang Groq |

> **Giải thích nhanh các metric:**
> - **Out TPS (`tok/s`):** Tổng tốc độ sinh token toàn hệ thống.
> - **TTFT (`ms`):** Time To First Token — thời gian chờ token đầu tiên (cảm giác phản hồi nhanh/chậm).
> - **TPOT (`ms`):** Time Per Output Token — độ trễ trung bình mỗi token sinh ra cho 1 user (`1000 / TPOT` = tốc độ đọc `tok/s/user`, ngưỡng đọc mượt ≥ 20 tok/s tương đương TPOT ≤ 50ms).
> - **E2E (`ms`):** Tổng thời gian hoàn tất toàn bộ câu trả lời.
> - **Preempt:** Số lần vLLM phải thu hồi/tính lại KV Cache do thiếu VRAM.

---

### 2. So sánh Direct vLLM (`:8000`) vs LiteLLM Proxy (`:4000`)

Cấu hình chuẩn (**Config A**: `--gpu-memory-utilization 0.90 --max-num-seqs 64 --max-model-len 8192`, KV Cache `46,080 tokens`), workload **512 input / 256 output tokens**:

| Concurrency | Đường gọi | Req/s | Out TPS (tok/s) | TTFT P50 / P95 (ms) | TPOT P50 / P95 (ms) | Tốc độ/user (tok/s) | E2E P50 / P95 (ms) | Preempt | Overhead LiteLLM |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| **1** | vLLM direct | 0.35 | 90.5 | 187.2 / 189.6 | 10.4 / 10.5 | 96.2 | 2,841 / 2,853 | 0 | — |
| **1** | LiteLLM | 0.35 | 89.0 | 204.5 / 209.6 | 10.5 / 10.6 | 95.2 | 2,875 / 2,901 | 0 | -1.7% TPS, +17ms TTFT |
| **4** | vLLM direct | 1.07 | 273.6 | 622.0 / 727.8 | 12.0 / 14.0 | 83.3 | 3,755 / 3,826 | 0 | — |
| **4** | LiteLLM | 1.05 | 269.5 | 696.7 / 745.1 | 12.0 / 14.1 | 83.3 | 3,763 / 3,843 | 0 | -1.5% TPS, +75ms TTFT |
| **8** | vLLM direct | 1.65 | 422.3 | 987.8 / 1,359.7 | 15.0 / 18.3 | 66.7 | 4,878 / 4,944 | 0 | — |
| **8** | LiteLLM | 1.63 | 416.0 | 1,091.1 / 1,351.1 | 14.5 / 18.3 | 69.0 | 4,800 / 4,933 | 0 | -1.5% TPS, +103ms TTFT |
| **16** | vLLM direct | 2.24 | 573.9 | 1,696.1 / 2,666.6 | 20.9 / 25.9 | 47.8 | 7,104 / 7,737 | 0 | — |
| **16** | LiteLLM | 2.23 | 570.5 | 1,866.1 / 2,651.4 | 21.0 / 25.6 | 47.6 | 6,930 / 7,245 | 0 | -0.6% TPS, +170ms TTFT |
| **32** | vLLM direct | 2.53 | **646.8** | 2,227.1 / 5,283.5 | 40.3 / 47.9 | 24.8 | 12,634 / 15,732 | 0 | — |
| **32** | LiteLLM | 2.44 | **625.3** | 2,915.2 / 5,218.6 | 38.4 / 47.7 | 26.0 | 12,799 / 14,842 | 0 | -3.3% TPS, +688ms TTFT |
| **64** | vLLM direct | 2.33 | 597.7 | 4,123.6 / 11,159.1 | 91.4 / 94.4 | 10.9 | 26,379 / 33,959 | **8** | — |
| **64** | LiteLLM | 2.32 | 593.8 | 4,050.8 / 11,295.1 | 91.4 / 94.1 | 10.9 | 26,439 / 33,946 | **8** | -0.7% TPS (bão hòa KV) |

**Nhận xét lớp LiteLLM Proxy:**
- Chi phí (overhead) của LiteLLM rất nhỏ: **giảm chỉ ~0.6%–3.3% throughput**, gần như **không ảnh hưởng TPOT** (tốc độ đọc của user giữ nguyên), chỉ tăng nhẹ TTFT ở mức `17–170ms` trong dải tải tối ưu (`C=1..16`).
- Tại **`C=64`** (`64 × 768 = 49,152 tokens > 46,080 tokens` KV Cache), hệ thống bị **8 preemptions**, làm throughput tụt ngược so với `C=32` và TPOT tăng gấp đôi (`91.4ms`).

---

### 3. Hiệu năng theo độ dài Context (Workload Profiles qua LiteLLM)

| Workload (In / Out) | Concurrency | Req/s | Out TPS (tok/s) | Total TPS (tok/s) | TTFT P50 / P95 (ms) | TPOT P50 / P95 (ms) | Tốc độ/user (tok/s) | E2E P50 / P95 (ms) | Preempt |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short Chat** (`128 / 128`) | 1 | 0.74 | 94.4 | 197.6 | 77.3 / 79.7 | 10.0 / 10.1 | 100.0 | 1,350 / 1,366 | 0 |
| **Short Chat** (`128 / 128`) | 8 | 4.19 | 536.3 | 1,122.9 | 358.3 / 427.3 | 11.7 / 13.6 | 85.5 | 1,832 / 1,908 | 0 |
| **Short Chat** (`128 / 128`) | 16 | 6.32 | 809.6 | 1,695.1 | 585.4 / 809.9 | 14.2 / 17.3 | 70.4 | 2,434 / 2,563 | 0 |
| **Short Chat** (`128 / 128`) | 32 | 7.48 | 956.8 | 2,003.4 | 991.5 / 1,532.3 | 22.5 / 27.8 | 44.4 | 4,001 / 4,173 | 0 |
| **Short Chat** (`128 / 128`) | 64 | 7.95 | **1,018.0** | 2,131.5 | 1,347.3 / 3,026.8 | 39.0 / 51.8 | 25.6 | 7,415 / 7,833 | 0 |
| **RAG Medium** (`2048 / 256`) | 1 | 0.27 | 69.1 | 625.6 | 824.4 / 841.6 | 11.7 / 11.7 | 85.5 | 3,803 / 3,829 | 0 |
| **RAG Medium** (`2048 / 256`) | 4 | 0.57 | 144.8 | 1,309.9 | 2,386.9 / 3,231.8 | 17.4 / 25.3 | 57.5 | 7,220 / 7,403 | 0 |
| **RAG Medium** (`2048 / 256`) | 8 | 0.68 | 175.1 | 1,584.1 | 4,061.4 / 6,330.0 | 31.1 / 42.9 | 32.2 | 11,675 / 13,508 | 0 |
| **RAG Medium** (`2048 / 256`) | 16 | 0.77 | **196.5** | 1,777.4 | 3,479.3 / 11,591.7 | 69.6 / 75.2 | 14.4 | 20,691 / 29,677 | 0 |
| **Long Context** (`4096 / 256`) | 1 | 0.20 | 52.1 | 887.6 | 1,787.2 / 1,865.3 | 13.4 / 13.5 | 74.6 | 5,207 / 5,287 | 0 |
| **Long Context** (`4096 / 256`) | 4 | 0.34 | 86.9 | 1,480.7 | 4,370.7 / 6,614.9 | 26.2 / 38.5 | 38.2 | 12,345 / 13,267 | 0 |
| **Long Context** (`4096 / 256`) | 8 | 0.37 | **93.8** | 1,598.6 | 7,999.4 / 11,791.8 | 56.4 / 73.6 | 17.7 | 21,773 / 25,540 | 0 |

---

### 4. So sánh các cấu hình Tuning vLLM Engine (qua LiteLLM)

| Config | Tham số vLLM | GPU KV Cache | Đặc tính |
| --- | --- | :---: | --- |
| **Config A (Gốc)** | `--gpu-memory-utilization 0.90 --max-num-seqs 64 --max-model-len 8192` | `46,080 tokens` | Throughput đỉnh cao nhất, dễ bị preemption khi burst `C=64` |
| **Config B (Low-Latency)** | `--gpu-memory-utilization 0.90 --max-num-seqs 32 --max-model-len 8192 --max-num-batched-tokens 1024` | `45,568 tokens` | Giảm sâu TTFT P50 (-20% đến -42%), **0 preemption** kể cả ở `C=64` |
| **Config C (4K / Large Batch)** | `--gpu-memory-utilization 0.92 --max-num-seqs 64 --max-model-len 4096 --max-num-batched-tokens 4096` | `44,544 tokens` | Batch prefill lớn tốn activation VRAM → KV cache ít hơn, kém Config A ở hầu hết mức tải |

| Kịch bản (LiteLLM) | Config | Out TPS (tok/s) | TTFT P50 / P95 (ms) | TPOT P50 / P95 (ms) | ITL P95 (ms) | E2E P50 / P95 (ms) | Preempt |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`512/256`, `C=8`** | **A** | 416.0 | 1,091.1 / 1,351.1 | 14.5 / 18.3 | 15.4 | 4,800 / 4,933 | 0 |
| **`512/256`, `C=8`** | **B** | **419.3** | **876.5** / **1,324.0** | 15.3 / **18.2** | **15.2** | 4,798 / **4,870** | 0 |
| **`512/256`, `C=8`** | **C** | 417.0 | 1,287.9 / 1,353.3 | **13.9** / 18.3 | 15.2 | **4,793** / 4,921 | 0 |
| **`512/256`, `C=16`** | **A** | **570.5** | 1,866.1 / **2,651.4** | **21.0** / **25.6** | **19.3** | **6,930** / **7,245** | 0 |
| **`512/256`, `C=16`** | **B** | 556.5 | **1,453.1** / 2,746.3 | 22.6 / 26.8 | 19.8 | 7,055 / 8,282 | 0 |
| **`512/256`, `C=16`** | **C** | 551.7 | 1,875.5 / 2,957.5 | 21.1 / 27.2 | 19.6 | 7,259 / 7,547 | 0 |
| **`512/256`, `C=32`** | **A** | **625.3** | 2,915.2 / **5,218.6** | **38.4** / **47.7** | **33.3** | **12,799** / 14,842 | 0 |
| **`512/256`, `C=32`** | **B** | 587.9 | **1,679.0** / 5,545.2 | 47.4 / 50.9 | 60.9 | 13,627 / 17,871 | 0 |
| **`512/256`, `C=32`** | **C** | 593.7 | 3,216.2 / 5,735.9 | 38.5 / 49.4 | 34.4 | 13,369 / **13,793** | 0 |
| **`512/256`, `C=64`** | **A** | **593.8** | **4,050.8** / **11,295.1** | 91.4 / 94.1 | 99.7 | 26,439 / 33,946 | 8 |
| **`512/256`, `C=64`** | **B** | 573.0 | 15,558.1 / 19,673.3 | **51.0** / **54.6** | 271.6 | 26,389 / **33,231** | **0** |
| **`512/256`, `C=64`** | **C** | 560.9 | 5,874.6 / 12,144.1 | 89.0 / 98.6 | **60.1** | **25,918** / 37,295 | 11 |
| **`2048/256`, `C=8`** | **A** | **175.1** | 4,061.4 / 6,330.0 | 31.1 / 42.9 | 23.7 | 11,675 / 13,508 | 0 |
| **`2048/256`, `C=8`** | **B** | 171.2 | **3,019.9** / **6,150.7** | 36.0 / **42.3** | 23.7 | **11,541** / 15,469 | 0 |
| **`2048/256`, `C=8`** | **C** | 173.5 | 4,403.6 / 6,568.1 | **28.0** / 43.7 | **23.2** | 12,118 / **12,205** | 0 |

---

### 5. Tổng hợp ngưỡng tối ưu (Sweet Spot) cho RTX 3060 12GB (`Qwen/Qwen3-4B-AWQ`)

| Bài toán / Workload | Ngưỡng Concurrency tối ưu | Trần khuyến nghị (tránh suy giảm) | Hiệu năng kỳ vọng tại ngưỡng tối ưu | Config khuyến nghị |
| --- | :---: | :---: | --- | --- |
| **Chat ngắn (`~128/128`)** | `16 – 32` | `64` | `810 – 957 tok/s` · TTFT `0.6 – 1.0s` · `44 – 70 tok/s/user` | **Config A** |
| **Chat / QA tiêu chuẩn (`~512/256`)** | `8 – 16` | `32` | `416 – 571 tok/s` · TTFT `1.1 – 1.9s` · `48 – 69 tok/s/user` | **Config A** (ưu tiên TPS) hoặc **Config B** (ưu tiên TTFT thấp & chống preemption) |
| **RAG trung bình (`~2048/256`)** | `4 – 8` | `8` | `145 – 175 tok/s` · TTFT `2.4 – 4.1s` · `32 – 58 tok/s/user` | **Config B** (giảm TTFT P50 ~26%) |
| **Long Context (`~4096/256`)** | `1 – 4` | `6` | `52 – 87 tok/s` · TTFT `1.8 – 4.4s` · `38 – 75 tok/s/user` | **Config A/B** (`max-model-len 8192`) |

**Khuyến nghị tinh chỉnh khi đưa vào sử dụng thực tế:**
1. **Không nên tăng `--max-num-batched-tokens 4096` (Config C)** trên RTX 3060 12GB vì tốn thêm bộ nhớ activation làm thu hẹp KV Cache và tăng TTFT.
2. Nếu ứng dụng chịu tải đột biến (burst > 32 request đồng thời) trên context `512–2048`, nên đặt **`--max-num-seqs 32`** (hoặc cấu hình rate limit/concurrency ở LiteLLM) để loại bỏ hoàn toàn preemption và giữ tốc độ đọc mượt cho các phiên đang chạy.
3. Nếu ưu tiên **TTFT thấp nhất cho giao diện chat**, dùng **`--max-num-batched-tokens 1024`** (Config B) giúp giảm **20%–42% TTFT P50** với đánh đổi rất nhỏ (~2–6% tổng throughput).
4. Cấu hình **`timeout: 60`** trong `litellm/config.yaml` đảm bảo không kích hoạt fallback ngoài ý muốn trong toàn bộ dải concurrency đến `64`, đồng thời vẫn bảo vệ dịch vụ khi `vllm` gặp sự cố.
