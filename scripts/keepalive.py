"""Keepalive for the TaxRAG Streamlit Community Cloud deployment.

Streamlit Cloud hibernates an app after 12 hours with no traffic. A plain
HTTP GET doesn't count -- Streamlit serves a static HTML shell to any GET
and returns 200 regardless of whether the app is asleep; the Python app
only starts once a browser runs the shell's JS and opens a WebSocket to
/_stcore/stream. This drives a real headless browser so that WebSocket
connection actually happens, and clicks the "wake up" button if the app
has gone to sleep.

Run manually: python scripts/keepalive.py (after `playwright install chromium`)
Run in CI: .github/workflows/keepalive.yml, every 6 hours.
"""

import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = "https://taxrag.streamlit.app/"

_WAKE_BUTTON_RE = re.compile("get this app back up", re.IGNORECASE)
# Matches app.py's st.title() -- only rendered once TaxRAG's own script has
# actually run, unlike the wake button, which disappears the instant the
# click registers, well before the app behind it is up.
_APP_HEADING_RE = re.compile("TaxRAG", re.IGNORECASE)
_WEBSOCKET_SETTLE_MS = 5_000
_WAKE_TIMEOUT_MS = 120_000


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(APP_URL, wait_until="networkidle")
        # networkidle fires once the static shell has loaded -- it doesn't
        # guarantee the shell's JS has finished opening the WebSocket that
        # actually counts as traffic, so wait a little longer for that.
        page.wait_for_timeout(_WEBSOCKET_SETTLE_MS)

        wake_button = page.get_by_role("button", name=_WAKE_BUTTON_RE)
        if wake_button.count() > 0:
            t0 = time.monotonic()
            wake_button.first.click()
            # A cron run every 6 hours should mean the app is never asleep
            # in practice -- this is a recovery path, not the common case.
            # The click itself already registered as traffic, which is the
            # entire point of this script, so a slow wake past the timeout
            # is logged, not treated as failure.
            try:
                page.get_by_role("heading", name=_APP_HEADING_RE).wait_for(
                    state="visible", timeout=_WAKE_TIMEOUT_MS
                )
            except PlaywrightTimeoutError:
                elapsed = time.monotonic() - t0
                print(f"WAKE {APP_URL}: app not up after {elapsed:.0f}s, giving up")
            else:
                elapsed = time.monotonic() - t0
                print(f"WAKE {APP_URL} ({elapsed:.0f}s)")
        else:
            print(f"OK {APP_URL}")

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {APP_URL}: {exc}", file=sys.stderr)
        sys.exit(1)
