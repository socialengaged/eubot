#!/usr/bin/env python3
import json
import urllib.request

def main() -> None:
    data = json.dumps(
        {"model": "eurobot-baby", "messages": [{"role": "user", "content": "hello"}]}
    ).encode()
    r = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    print(urllib.request.urlopen(r, timeout=400).read().decode())


if __name__ == "__main__":
    main()
