"""Smoke-test the running server against the Ponsse index.

Usage:
  python scripts/smoke_test.py --url http://127.0.0.1:8000
"""

import argparse
import json
import sys
import urllib.request


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--query",
        default="How do I replace the hydraulic filter on the Scorpion King?",
    )
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    print(f"[1/4] GET  {args.url}/health")
    print(json.dumps(_get(args.url + "/health"), indent=2))

    print(f"\n[2/4] GET  {args.url}/stats")
    print(json.dumps(_get(args.url + "/stats"), indent=2))

    print(f"\n[3/4] GET  {args.url}/v1/models")
    print(json.dumps(_get(args.url + "/v1/models"), indent=2))

    print(f"\n[4/4] POST {args.url}/chat  (query={args.query!r})")
    if args.skip_generation:
        # Use /query with generate_answer=False for retrieval-only sanity check
        resp = _post(
            args.url + "/query",
            {"query": args.query, "top_k": 5, "generate_answer": False},
        )
        print(f"  retrieval_ms: {resp['retrieval_latency_ms']:.1f}")
        print("  top pages:")
        for p in resp["retrieved_pages"]:
            print(f"    {p['score']:.3f}  {p['page_id']}")
    else:
        resp = _post(
            args.url + "/chat",
            {
                "messages": [{"role": "user", "content": args.query}],
                "provider": "aiwo-rag",
                "model": "aiwo-rag",
                "top_k": 5,
                "max_pages_for_vlm": 3,
            },
        )
        print(f"  retrieval_ms: {resp['retrieval_latency_ms']:.1f}")
        print(f"  total_ms:     {resp['total_latency_ms']:.1f}")
        print("  top pages:")
        for p in resp["retrieved_pages"]:
            print(f"    {p['score']:.3f}  {p['page_id']}")
        print("\n  answer:")
        print("  " + resp["result"]["content"].replace("\n", "\n  "))


if __name__ == "__main__":
    sys.exit(main())
