#!/usr/bin/env python3
import sys

p = sys.argv[1]
with open(p, "rb") as f:
    d = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"")
with open(p, "wb") as f:
    f.write(d)
