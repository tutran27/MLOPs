# vLLM & AI Gateway Practice Stack

Dự án thực hành triển khai mô hình ngôn ngữ lớn (LLM) với **vLLM**, quản lý và điều phối API tập trung qua **LiteLLM Gateway**, cùng hệ thống giám sát toàn diện (**Prometheus, Grafana, Loki, Alloy**).

---

## 1. Sơ đồ kiến trúc hệ thống (System Architecture)

```mermaid
flowchart TB
    subgraph Clients["Clients / Consumers"]
        ClientApp["Client App / Web / Chatbot"]
        Bruno["Bruno / Postman"]
        DevSDK["Python SDK (openai)"]
    end

    subgraph Gateway["AI Gateway & API Governance"]
        LiteLLM["LiteLLM Proxy\n(Port: 4000)"]
        LiteLLM_DB[("PostgreSQL\n(litellm_db:5432)\n- Virtual Keys\n- Users & Teams\n- Budgets & Quotas")]
        LiteLLM UI["Admin UI\n(/ui)"]
        
        LiteLLM <--> LiteLLM_DB
        LiteLLM UI --- LiteLLM
    end

    subgraph LLM_Providers["LLM Inference Providers"]
        vLLM["Local vLLM Engine\n(Port: 8000 / GPU)\nModel: qwen3-4b"]
        Groq["Cloud External Provider\n(Groq API)\nModel: openai/gpt-oss-20b"]
    end

    subgraph Observability["Observability & Monitoring Stack"]
        Prometheus["Prometheus\n(Port: 9090)\nTime-series Metrics"]
        Alloy["Grafana Alloy\nLog Collector"]
        Loki["Grafana Loki\n(Port: 3100)\nLog Storage"]
        Grafana["Grafana Dashboards\n(Port: 3000)"]

        Prometheus --> Grafana
        Loki --> Grafana
        Alloy --> Loki
    end

    %% Client traffic
    ClientApp -->|"API Key: Bearer sk-..."| LiteLLM
    Bruno -->|"API Key: Bearer sk-..."| LiteLLM
    DevSDK -->|"API Key: Bearer sk-..."| LiteLLM

    %% Routing
    LiteLLM -->|"Load balancing / Route"| vLLM
    LiteLLM -->|"Route (External API)"| Groq

    %% Metrics & Logs Scrape
    vLLM -.->|"Metrics (:8000/metrics)"| Prometheus
    LiteLLM -.->|"Metrics (:4000/metrics)"| Prometheus
    vLLM -.->|"Docker stdout logs"| Alloy
    LiteLLM -.->|"Docker stdout logs"| Alloy
```

---

## 2. Cấu trúc thư mục (Directory Structure)

```text
vLLM/
├── README.md
├── requirements.txt
├── demo.py                          # Script test trực tiếp OpenAI SDK / Groq
├── tokenization.py                  # Lab tìm hiểu về Tokenizer
├── lesson_1_2/                      # Thực hành cơ bản vLLM
├── lesson_1_3/                      # Thực hành streaming, non-streaming, generation config
├── lesson_1_4/                      # Load testing vLLM server
│   └── load_test.py
└── vllm-stack/                      # Stack triển khai sản xuất hoàn chỉnh (Docker Compose)
    ├── .env                         # Khai báo cấu hình, API keys, Model tên...
    ├── compose.yaml                 # Docker Compose orchestrate toàn bộ stack
    ├── litellm/
    │   └── config.yaml              # Cấu hình routing model, master key, Postgres cho LiteLLM
    ├── prometheus/
    │   └── prometheus.yml           # Khai báo targets scrape metrics (vLLM & LiteLLM)
    ├── loki/
    │   └── loki.yaml                # Cấu hình lưu trữ log cho Loki
    ├── alloy/
    │   └── config.alloy             # Cấu hình thu thập log từ Docker gửi về Loki
    └── grafana/
        └── provisioning/            # Tự động nạp datasources (Loki & Prometheus) vào Grafana
            └── datasources/
                ├── loki.yaml
                └── prometheus.yaml
```

---

## 3. Thành phần và Cổng dịch vụ (Service Ports)

| Dịch vụ | Cổng (Host:Container) | Chức năng chính |
| :--- | :--- | :--- |
| **LiteLLM Gateway** | `4000:4000` | Cổng API tập trung, quản lý Virtual Key, Rate limit, Routing, Admin UI (`/ui`) |
| **vLLM Engine** | Nội bộ (`8000`) | Engine tăng tốc suy luận LLM bằng GPU (phục vụ model `qwen3-4b`) |
| **PostgreSQL** | Nội bộ (`5432`) | Cơ sở dữ liệu lưu trữ cấu hình, API key, thông tin người dùng, lịch sử chi tiêu cho LiteLLM |
| **Grafana** | `3000:3000` | Giao diện trực quan hóa Dashboard giám sát Metrics & Logs (user: `admin` / pass: `admin123`) |
| **Prometheus** | `9090:9090` | Thu thập metrics định kỳ từ vLLM (`:8000`) và LiteLLM (`:4000`) |
| **Loki** | `3100:3100` | Lưu trữ tập trung và đánh chỉ mục logs |
| **Alloy** | N/A | Agent gom Docker logs đẩy sang Loki |

---

## 4. Hướng dẫn triển khai Stack (`vllm-stack`)

### Bước 1: Chuẩn bị biến môi trường
Tạo hoặc chỉnh sửa file `vllm-stack/.env`:

```env
MODEL=RedHatAI/Qwen3.5-4B-quantized.w4a16
MODEL_SERVED=qwen3-4b
HF_TOKEN=hf_your_huggingface_token
GRAFANA_ADMIN_PASSWORD=admin123
GROQ_API_KEY=gsk_your_groq_api_key
LITELLM_MASTER_KEY=sk-your-master-key
LITELLM_UI_PASSWORD=admin123
```

### Bước 2: Khởi chạy toàn bộ stack với Docker Compose

```bash
cd vllm-stack
docker compose up -d
```

### Bước 3: Quản trị và kiểm soát API qua LiteLLM UI
1. Truy cập Web UI tại: `http://localhost:4000/ui`
2. Đăng nhập bằng `admin` và mật khẩu cấu hình.
3. Tạo **Virtual Key** (`sk-...`):
   - Đặt giới hạn **Max Budget** (ngân sách chi tiêu $).
   - Đặt giới hạn tần suất **RPM / TPM** (Requests / Tokens per minute).
   - Chọn danh sách **Allowed Models** được phép truy cập (`qwen3-4b` hoặc `openai/gpt-oss-20b`).

### Bước 4: Test gửi Request qua LiteLLM Gateway
Dùng `curl` hoặc Bruno/Postman gửi request tới endpoint LiteLLM:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <VIRTUAL_KEY_CUA_BAN>" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "user", "content": "Hello, introduce yourself!"}
    ]
  }'
```

---

## 5. Lab thực hành cơ bản (Standalone vLLM)

### Setup môi trường Python cục bộ

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy trực tiếp vLLM Server qua CLI (Lesson 1.3)

```bash
VLLM_LOGGING_LEVEL=INFO \
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-0.6b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm \
  --enable-request-id-headers \
  --enable-log-requests
```

#### Test nhanh lệnh curl:
* **Kiểm tra Health**: `curl -i http://127.0.0.1:8000/health`
* **Danh sách Models**: `curl -sS http://127.0.0.1:8000/v1/models | jq`
* **Kiểm tra Metrics**: `curl -sS http://127.0.0.1:8000/metrics | grep 'vllm:num_requests'`
