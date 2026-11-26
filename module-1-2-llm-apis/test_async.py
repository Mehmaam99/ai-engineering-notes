import asyncio
import time


async def fake_llm_call(name: str) -> str:
    print(f"Start {name}")
    await asyncio.sleep(2)
    print(f"Done {name}")
    return name


async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        fake_llm_call("User 1"),
        fake_llm_call("User 2"),
        fake_llm_call("User 3"),
    )
    elapsed = time.perf_counter() - start
    print(f"Finished {results} in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
