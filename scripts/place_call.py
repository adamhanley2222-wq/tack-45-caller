#!/usr/bin/env python3
"""Place one outbound call through a running tack45-caller service.

This talks to your service, which talks to Twilio. It places a real call when
the service has live Twilio credentials and the destination is inside the
calling window. Nothing in the test suite runs this script.

Example:
    CALLER_BASE_URL=http://127.0.0.1:8787 API_BEARER_TOKEN=... \\
        python scripts/place_call.py --to +61412345678 --name "Sam Lee" \\
        --business "Harbour Roasters" --notes "Asked about invoices"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Place one Tack45 outbound call.")
    parser.add_argument("--to", required=True, help="Destination number, E.164 or Australian national.")
    parser.add_argument("--name", default="", help="Person you expect to reach.")
    parser.add_argument("--business", default="", help="Business name.")
    parser.add_argument("--notes", default="", help="Context for the model.")
    parser.add_argument("--timezone", default=None, help="IANA timezone. Default is inferred, else Australia/Sydney.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CALLER_BASE_URL", "http://127.0.0.1:8787"),
        help="Where the tack45-caller HTTP API is listening.",
    )
    args = parser.parse_args()

    token = os.environ.get("API_BEARER_TOKEN", "")
    if not token:
        print("API_BEARER_TOKEN is not set.", file=sys.stderr)
        return 2

    payload = {
        "to": args.to,
        "name": args.name,
        "business": args.business,
        "notes": args.notes,
    }
    if args.timezone:
        payload["timezone"] = args.timezone

    url = args.base_url.rstrip("/") + "/calls"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    print(
        f"Requesting a call to {args.to} via {url}. This dials for real if the service is configured.",
        file=sys.stderr,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            print(body)
            return 0 if 200 <= response.status < 300 else 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(detail or exc.reason, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
