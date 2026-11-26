# Day 1: LLM Fundamentals & FastAPI Async Basics (26 Nov 2026)

## 1. Tokenization (BPE — Byte Pair Encoding)
- **What is a Token?** LLMs cannot read English or Urdu letters directly—they only process numbers. Tokenization splits text into small pieces (whole words, sub-words, or symbols) and maps each piece to a numeric ID in the model's vocabulary.
- **Rule of Thumb:** `1 token ≈ 4 characters` or `0.75 English words` (so `100 tokens ≈ 75 words`, and `1,000 tokens ≈ 750 words`).
  - **Practical Cost Math:** If you summarize 10,000 reviews of 300 words each: `300 / 0.75 = 400 tokens/review` → `4M input tokens`.
- **Interview Q:** *Why do JSON and non-English languages (like Roman Urdu) cost more tokens?*
  - **Answer:** Formatting characters (`{`, `}`, `"`, `:`) and less-common words split into smaller sub-word/character tokens, increasing token count, latency, and API cost.

---

## 2. Context Window & "Lost in the Middle"
- **What is Context Window?** The maximum number of tokens an LLM can process in a single request—this includes **both Input (prompt + history + RAG docs) and Output (completion)**.
- **Interview Q:** *Why shouldn't we dump an entire database into the prompt even if the context window is 128k+?*
  - **Answer:** (1) Higher token cost, (2) Slower Time-to-First-Token (TTFT), and (3) **Lost in the Middle**—LLMs pay strong attention to the start and end of a prompt, often missing details buried in the middle.

---

## 3. How LLMs Pick the Next Word: Logits, Temperature, Softmax & Top-P
When predicting the next token (e.g., `"The sky is ___"`), an LLM runs a 4-step pipeline:

1. **Logits (Raw Scores):** The model outputs a raw, unnormalized score for every token in its vocabulary (e.g., `"blue": 5.0`, `"clear": 3.0`, `"green": 1.0`).
2. **Temperature ($T$) — Scaling the Logits:** Before computing probabilities, every logit is divided by $T$ ($\text{logit} / T$):
   - **Low $T$ (e.g., `0.1`):** Dividing by `0.1` widens the gap (`50` vs `30`), so the top word dominates → **deterministic, factual** (best for RAG, SQL, JSON extraction). At `T = 0`, it always picks the highest logit (*greedy decoding*).
   - **High $T$ (e.g., `1.5`):** Shrinks the gap (`3.3` vs `2.0`), giving lower-ranked words a chance → **creative, random**.
3. **Softmax (Logits → Probabilities):** Converts scaled logits into probabilities that sum to `1.0` (100%) using $e^{x_i} / \sum e^{x_j}$ (e.g., `"blue": 0.84`, `"clear": 0.11`, `"green": 0.05`).
4. **Top-P (Nucleus Sampling):** Sorts tokens by probability and keeps only the smallest set whose cumulative probability is `<= P`. With `top_p = 0.90`, `"blue"` (`0.84`) + `"clear"` (`0.11`) = `0.95 >= 0.90`, so `"green"` (`0.05`) is discarded.
- **Interview Rule:** Tune **either** Temperature or Top-P, not both together. For production/RAG, use `temperature=0, top_p=1`.

---

## 4. Latency Metrics: TTFT vs TPOT
- **TTFT (Time to First Token):** How long the user waits after hitting Send before seeing the first word.
- **TPOT (Time Per Output Token):** Speed of generating each subsequent token.
- **Interview Q:** *Does streaming make an LLM faster?*
  - **Answer:** Total generation time is the same, but streaming drastically lowers **TTFT**, making the app feel instant to users.

---

## 5. `asyncio` & FastAPI Concurrency
- **The Restaurant Waiter Analogy:**
  - **Synchronous (`def` + `requests.post`):** One waiter takes Order #1, walks into the kitchen, and stands staring at the stove for 5 seconds until it's cooked. Everyone else waits. In FastAPI, a blocking call freezes the entire event loop.
  - **Asynchronous (`async def` + `await`):** The waiter hands Order #1 to the kitchen (`await`) and immediately takes Order #2 from the next table while #1 is cooking.
- **4 Core Building Blocks:**
  1. **Event Loop:** Single-threaded manager that switches between tasks during network/disk I/O waits.
  2. **`async def`:** Defines a coroutine that can pause midway.
  3. **`await`:** Pauses the current function during a slow I/O call (like `AsyncGroq` or PostgreSQL) so other requests can run.
  4. **`asyncio.gather()`:** Runs multiple `async` tasks concurrently (e.g., 3 calls of 2s each finish in ~2s total instead of 6s).
- **Golden Rule for FastAPI:** Never use blocking calls (`time.sleep()`, `requests.get()`) inside an `async def` route—always use `await asyncio.sleep()`, `httpx.AsyncClient`, or `AsyncGroq`/`AsyncOpenAI`.

```python
import asyncio, time

async def fake_llm_call(name: str):
    await asyncio.sleep(2)
    return name

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(
        fake_llm_call("User 1"),
        fake_llm_call("User 2"),
        fake_llm_call("User 3"),
    )
    print(f"Finished {results} in {time.perf_counter() - start:.2f}s")

asyncio.run(main())
```
