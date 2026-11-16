import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-123",
)


async def run_request(request_id: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        start = time.perf_counter()

        stream = await client.chat.completions.create(
            model="qwen3-4b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Bạn là Kỹ sư Trưởng Hệ thống AI. Hãy phân tích bản đặc tả hệ thống và viết một tài liệu thiết kế kỹ thuật chi tiết:\n\n"
                        "1. Bối cảnh: Hệ thống phục vụ 100.000 người dùng trực tuyến, đang gặp hiện tượng TTFT tăng cao và KV Cache cạn kiệt vào giờ cao điểm.\n"
                        "2. Yêu cầu phân tích: So sánh Prefill Phase và Decode Phase; phân tích cơ chế Continuous Batching và PagedAttention trong việc tối ưu bộ nhớ; "
                        "so sánh hai cơ chế xử lý khi thiếu VRAM là Swapping và Preemption.\n"
                        "3. Giám sát hệ thống: Đưa ra danh sách các metrics quan trọng cần theo dõi trên Prometheus/Grafana (Running/Waiting requests, "
                        "GPU cache usage, TTFT, TPOT, Throughput) kèm theo công thức PromQL và ngưỡng cảnh báo cụ thể.\n"
                        "4. Kiến trúc mở rộng: Đề xuất mô hình kết hợp Tensor Parallelism, Pipeline Parallelism và giải pháp định tuyến tải.\n\n"
                        "Hãy viết câu trả lời thật chi tiết, phân tích sâu về mặt kỹ thuật, có số liệu minh họa và lập luận rõ ràng từng mục."
                    ),
                }
            ],
            temperature=0,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers={"X-Request-Id": f"load-{request_id}"},
        )

        ttft = None
        usage = None

        async for chunk in stream:
            now = time.perf_counter()
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content and ttft is None:
                    ttft = now - start

            if chunk.usage is not None:
                usage = chunk.usage

        end = time.perf_counter()
        e2e = end - start

        output_tokens = usage.completion_tokens if usage is not None else 0
        tpot = (e2e - ttft) / (output_tokens - 1) if ttft is not None and output_tokens > 1 else None

        return {
            "request_id": request_id,
            "ttft": ttft,
            "e2e": e2e,
            "tpot": tpot,
            "output_tokens": output_tokens,
        }


def percentile(values, p):
    values = sorted(values)
    index = int((len(values) - 1) * p)
    return values[index]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    benchmark_start = time.perf_counter()

    tasks = [run_request(i, semaphore) for i in range(args.requests)]
    results = await asyncio.gather(*tasks)

    benchmark_end = time.perf_counter()
    duration = benchmark_end - benchmark_start

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if r["tpot"] is not None]
    e2es = [r["e2e"] for r in results]
    total_output_tokens = sum(r["output_tokens"] for r in results)

    print("\n===== RESULT =====")
    print(f"Requests: {len(results)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Duration: {duration:.2f}s")
    print(f"Request throughput: {len(results) / duration:.2f} req/s")
    print(f"Output throughput: {total_output_tokens / duration:.2f} tokens/s\n")

    print(f"Mean TTFT: {statistics.mean(ttfts):.3f}s")
    print(f"P95 TTFT: {percentile(ttfts, 0.95):.3f}s")
    print(f"Mean TPOT: {statistics.mean(tpots):.3f}s/token")
    print(f"P95 TPOT: {percentile(tpots, 0.95):.3f}s/token")
    print(f"Mean E2E: {statistics.mean(e2es):.3f}s")
    print(f"P95 E2E: {percentile(e2es, 0.95):.3f}s")


asyncio.run(main())