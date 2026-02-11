"""
Benchmark CosyVoice API server: batch_size=1 (sequential) and batch_size=2 (concurrent).
"""
import asyncio
import time
import struct
import httpx
import sys

SERVER_URL = "http://localhost:8005"
VOICE = "Professor Allen Yang"

# Test sentences of varying length
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming how we interact with technology in our daily lives.",
    "Welcome to today's lecture on machine learning. We will cover neural networks, backpropagation, and gradient descent.",
    "The weather forecast predicts sunny skies with a high of seventy five degrees for the rest of the week.",
    "In computer science, algorithms are step by step procedures for solving problems efficiently and correctly.",
]

async def single_request(client: httpx.AsyncClient, text: str, request_id: int):
    """Send a single TTS request and measure timing."""
    payload = {
        "model": "cosyvoice",
        "input": text,
        "voice": VOICE,
        "stream": False,
        "response_format": "pcm",
    }
    start = time.perf_counter()
    resp = await client.post(f"{SERVER_URL}/v1/audio/speech", json=payload)
    elapsed = time.perf_counter() - start

    if resp.status_code != 200:
        print(f"  [req {request_id}] ERROR {resp.status_code}: {resp.text[:200]}")
        return None

    pcm_bytes = resp.content
    # PCM s16le mono @ 24kHz
    num_samples = len(pcm_bytes) // 2
    audio_duration = num_samples / 24000.0
    rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')

    print(f"  [req {request_id}] text_len={len(text):3d}, audio={audio_duration:.2f}s, time={elapsed:.2f}s, RTF={rtf:.3f}")
    return {"audio_duration": audio_duration, "elapsed": elapsed, "rtf": rtf, "text_len": len(text)}


async def benchmark_sequential(n_runs=5):
    """batch_size=1: sequential requests."""
    print(f"\n{'='*60}")
    print(f"BATCH SIZE = 1 (Sequential, {n_runs} requests)")
    print(f"{'='*60}")

    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(n_runs):
            text = SENTENCES[i % len(SENTENCES)]
            r = await single_request(client, text, i + 1)
            if r:
                results.append(r)

    if results:
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        avg_audio = sum(r["audio_duration"] for r in results) / len(results)
        print(f"\n  Summary ({len(results)} requests):")
        print(f"    Avg RTF:            {avg_rtf:.3f}")
        print(f"    Avg latency:        {avg_time:.2f}s")
        print(f"    Avg audio duration: {avg_audio:.2f}s")
    return results


async def benchmark_concurrent(n_concurrent=2, n_rounds=3):
    """batch_size=2: concurrent requests."""
    print(f"\n{'='*60}")
    print(f"BATCH SIZE = {n_concurrent} (Concurrent, {n_rounds} rounds)")
    print(f"{'='*60}")

    all_results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for round_idx in range(n_rounds):
            print(f"\n  --- Round {round_idx + 1} ---")
            tasks = []
            for j in range(n_concurrent):
                idx = (round_idx * n_concurrent + j) % len(SENTENCES)
                text = SENTENCES[idx]
                tasks.append(single_request(client, text, j + 1))

            round_start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            round_elapsed = time.perf_counter() - round_start

            valid = [r for r in results if r is not None]
            if valid:
                total_audio = sum(r["audio_duration"] for r in valid)
                # Throughput RTF: total audio generated / wall clock time
                throughput_rtf = round_elapsed / total_audio * n_concurrent if total_audio > 0 else float('inf')
                print(f"  Round wall time: {round_elapsed:.2f}s, total audio: {total_audio:.2f}s, throughput RTF: {throughput_rtf:.3f}")
                all_results.extend(valid)

    if all_results:
        # Per-request RTF (each request measured individually)
        avg_rtf = sum(r["rtf"] for r in all_results) / len(all_results)
        avg_time = sum(r["elapsed"] for r in all_results) / len(all_results)
        avg_audio = sum(r["audio_duration"] for r in all_results) / len(all_results)
        total_audio = sum(r["audio_duration"] for r in all_results)
        print(f"\n  Summary ({len(all_results)} requests across {n_rounds} rounds):")
        print(f"    Avg per-request RTF:  {avg_rtf:.3f}")
        print(f"    Avg per-request time: {avg_time:.2f}s")
        print(f"    Avg audio duration:   {avg_audio:.2f}s")
    return all_results


async def main():
    # Warm up
    print("Warming up (1 request)...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await single_request(client, "Hello, this is a warmup test.", 0)

    seq_results = await benchmark_sequential(n_runs=5)
    conc_results = await benchmark_concurrent(n_concurrent=2, n_rounds=3)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    if seq_results:
        avg_seq = sum(r["rtf"] for r in seq_results) / len(seq_results)
        print(f"  batch_size=1 (sequential):  Avg RTF = {avg_seq:.3f}")
    if conc_results:
        avg_conc = sum(r["rtf"] for r in conc_results) / len(conc_results)
        print(f"  batch_size=2 (concurrent):  Avg RTF = {avg_conc:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
