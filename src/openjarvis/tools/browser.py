"""Browser automation tools — Playwright-based web interaction."""

from __future__ import annotations

import atexit
import base64
import logging
import queue
import re
import threading
import time
from concurrent.futures import Future
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)
_BROWSER_STOP = object()
# Leave time for completion and cleanup before the tool runner stops waiting.
_BROWSER_CLICK_WINDOW_SECONDS = 22.0
_MAX_CLICK_MATCHES = 100


class _ClickOutcomeUnknown(RuntimeError):
    """A click was attempted but the browser did not confirm the outcome."""


def _browser_request_ssrf_error(url: str, *, websocket: bool = False) -> str | None:
    """Fail closed unless a browser request targets a safe HTTP(S) host."""
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        if websocket:
            if scheme not in ("ws", "wss"):
                return f"Unsupported WebSocket scheme: {scheme or '(missing)'}"
            parsed = parsed._replace(scheme="https" if scheme == "wss" else "http")
        elif scheme not in ("http", "https"):
            return f"Unsupported browser request scheme: {scheme or '(missing)'}"
        from openjarvis.security.ssrf import check_ssrf

        return check_ssrf(urlunsplit(parsed))
    except Exception as exc:
        return f"SSRF guard failed: {type(exc).__name__}"


class _BrowserSession:
    """Manages a shared Playwright browser session (lazy init)."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._owner_thread_id: int | None = None
        self._closed = False

    def _worker_loop(self) -> None:
        self._owner_thread_id = threading.get_ident()
        while True:
            item = self._queue.get()
            try:
                if item is _BROWSER_STOP:
                    return
                future, callback = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(callback())
                except BaseException as exc:
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _submit(self, callback):
        if self._owner_thread_id == threading.get_ident():
            return callback()
        future: Future[Any] = Future()
        with self._worker_lock:
            if self._closed:
                raise RuntimeError("Browser session is closed.")
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="openjarvis-browser-session",
                    daemon=True,
                )
                self._worker.start()
            self._queue.put_nowait((future, callback))
        return future.result()

    def run(self, operation):
        """Run one browser operation on the thread that owns Playwright."""
        return self._submit(lambda: operation(self.page))

    def _guard_request(self, route: Any) -> None:
        url = str(route.request.url)
        reason = _browser_request_ssrf_error(url)
        if reason:
            logger.warning("Blocked browser request by SSRF guard: %s", reason)
            route.abort("blockedbyclient")
            return
        route.continue_()

    def _guard_websocket(self, websocket_route: Any) -> None:
        url = str(websocket_route.url)
        reason = _browser_request_ssrf_error(url, websocket=True)
        if reason:
            logger.warning("Blocked browser WebSocket by SSRF guard: %s", reason)
            websocket_route.close(
                code=1008,
                reason="Blocked by local browser safety policy",
            )
            return
        websocket_route.connect_to_server()

    def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright not installed. Install with: uv sync --extra browser"
            )
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=True)
            self._context = self._browser.new_context(service_workers="block")
            route_web_socket = getattr(self._context, "route_web_socket", None)
            if not callable(route_web_socket):
                raise RuntimeError(
                    "SSRF-safe browser routing requires Playwright 1.48 or newer. "
                    "Upgrade with: uv sync --extra browser"
                )
            self._context.route("**/*", self._guard_request)
            route_web_socket("**/*", self._guard_websocket)
            self._page = self._context.new_page()
        except BaseException:
            self._close_resources()
            raise

    @property
    def page(self):
        self._ensure_browser()
        return self._page

    def close(self) -> None:
        close_inline = False
        with self._worker_lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            if worker is None or not worker.is_alive():
                close_inline = True
            else:
                future: Future[Any] = Future()
                self._queue.put_nowait((future, self._close_resources))
                self._queue.put_nowait(_BROWSER_STOP)
        if close_inline:
            self._close_resources()
            return
        if worker.ident == threading.get_ident():
            return
        try:
            future.result(timeout=5)
        except Exception:
            logger.debug("Browser cleanup did not finish cleanly", exc_info=True)
        worker.join(timeout=5)
        if not worker.is_alive():
            # If the worker died while cleanup was queued, make a final
            # best-effort release and clear references instead of leaking state.
            self._close_resources()

    def _close_resources(self) -> None:
        context, browser, playwright = self._context, self._browser, self._playwright
        self._playwright = self._browser = self._context = self._page = None
        for resource in (context, browser, playwright):
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if not callable(close):
                close = getattr(resource, "stop", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close browser resource", exc_info=True)


_session = _BrowserSession()
atexit.register(_session.close)


def _run_browser_action(operation):
    """Use the owner thread in production and support simple test doubles."""
    session = _session
    if isinstance(session, _BrowserSession):
        return session.run(operation)
    return operation(session.page)


# ---------------------------------------------------------------------------
# Tool 1: BrowserNavigateTool
# ---------------------------------------------------------------------------


@ToolRegistry.register("browser_navigate")
class BrowserNavigateTool(BaseTool):
    """Navigate to a URL in the browser."""

    tool_id = "browser_navigate"
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_navigate",
            description=(
                "Navigate to a URL in the browser."
                " Returns the page title and text content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to.",
                    },
                    "wait_for": {
                        "type": "string",
                        "description": (
                            "Wait condition: 'load', 'domcontentloaded',"
                            " or 'networkidle'. Default: 'load'."
                        ),
                    },
                },
                "required": ["url"],
            },
            category="browser",
            required_capabilities=["network:fetch"],
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult(
                tool_name="browser_navigate",
                content="No URL provided.",
                success=False,
            )

        wait_for = params.get("wait_for", "load")
        if wait_for not in ("load", "domcontentloaded", "networkidle"):
            wait_for = "load"

        # Check the top-level URL before opening the browser, then the session
        # route checks every redirected, embedded, and WebSocket request too.
        ssrf_error = _browser_request_ssrf_error(url)
        if ssrf_error:
            return ToolResult(
                tool_name="browser_navigate",
                content=f"SSRF blocked: {ssrf_error}",
                success=False,
            )

        try:

            def navigate(page: Any) -> tuple[str, str, int | None]:
                response = page.goto(url, wait_until=wait_for)
                title = page.title()
                text_content = page.inner_text("body")
                if len(text_content) > 5000:
                    text_content = text_content[:5000] + "\n\n[Content truncated]"
                status = response.status if response else None
                return title, text_content, status

            title, text_content, status = _run_browser_action(navigate)
            return ToolResult(
                tool_name="browser_navigate",
                content=f"Title: {title}\n\n{text_content}",
                success=True,
                metadata={"url": url, "title": title, "status": status},
            )
        except ImportError:
            return ToolResult(
                tool_name="browser_navigate",
                content=(
                    "playwright not installed. Install with: uv sync --extra browser"
                ),
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_navigate",
                content=f"Navigation error: {exc}",
                success=False,
            )


# ---------------------------------------------------------------------------
# Tool 2: BrowserClickTool
# ---------------------------------------------------------------------------


@ToolRegistry.register("browser_click")
class BrowserClickTool(BaseTool):
    """Click an element on the page."""

    tool_id = "browser_click"
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_click",
            description=(
                "Click an element on the current page. Pass the text you see"
                " on the element (button label, link title), a CSS selector,"
                " or a short description like 'first video result'."
                " Use a precise selector if multiple visible elements match."
                " Inspect the page if the result is uncertain."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": (
                            "Visible text of the element, a CSS selector, or"
                            " a short description (e.g. 'first video result')."
                        ),
                    },
                    "by_text": {
                        "type": "boolean",
                        "description": (
                            "If true, click by text content"
                            " instead of CSS selector. Default: false."
                        ),
                    },
                    "target": {
                        "type": "string",
                        "description": "Alias for selector.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Alias for selector, for visible element text.",
                    },
                    "element": {
                        "type": "string",
                        "description": "Alias for selector.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Alias for selector.",
                    },
                },
                # Require one accepted spelling without forcing callers to
                # use ``selector`` when an advertised alias is supplied.
                "anyOf": [
                    {"required": ["selector"]},
                    {"required": ["target"]},
                    {"required": ["text"]},
                    {"required": ["element"]},
                    {"required": ["query"]},
                ],
            },
            category="browser",
            requires_confirmation=True,
        )

    _ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}

    # Filler words stripped from natural-language descriptions before
    # keyword matching ("the first video result" -> "video"). Derive the
    # ordinal entries from the supported mapping so adding one cannot leave it
    # behind as a literal role-name keyword.
    _DESC_STOPWORDS = frozenset(
        "the a an top main big red blue green"
        " result results item element link button option entry row".split()
    ) | frozenset(_ORDINALS)

    def execute(self, **params: Any) -> ToolResult:
        # Models routinely name this param `target`, `text`, `element` or
        # `query` instead of `selector`; rejecting those costs the whole
        # workflow, so accept the common aliases.
        selector = str(
            params.get("selector")
            or params.get("target")
            or params.get("text")
            or params.get("element")
            or params.get("query")
            or ""
        ).strip()
        if not selector:
            return ToolResult(
                tool_name="browser_click",
                content=(
                    "No selector provided. Pass the visible text of the"
                    " element or a CSS selector."
                ),
                success=False,
            )

        by_text = params.get("by_text", False)

        deadline = time.monotonic() + _BROWSER_CLICK_WINDOW_SECONDS
        try:
            strategy = _run_browser_action(
                lambda page: self._resolve_and_click(
                    page, selector, by_text, deadline=deadline
                )
            )
            return ToolResult(
                tool_name="browser_click",
                content=f"Clicked element: {selector}",
                success=True,
                metadata={
                    "selector": selector,
                    "by_text": by_text,
                    "strategy": strategy,
                },
            )
        except ImportError:
            return ToolResult(
                tool_name="browser_click",
                content=(
                    "playwright not installed. Install with: uv sync --extra browser"
                ),
                success=False,
            )
        except _ClickOutcomeUnknown as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Click error: {exc}",
                success=False,
                metadata={"outcome": "unknown"},
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_click",
                content=f"Click error: {exc}",
                success=False,
            )

    def _resolve_and_click(
        self,
        page: Any,
        selector: str,
        by_text: bool,
        *,
        deadline: float | None = None,
    ) -> str:
        """Choose one specific DOM element before committing a single click."""
        if deadline is None:
            deadline = time.monotonic() + _BROWSER_CLICK_WINDOW_SECONDS

        def time_remaining_ms() -> int:
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise RuntimeError("The click expired before any input was sent.")
            return remaining

        words = re.findall(r"\w+", selector.lower())
        ordinal = next((self._ORDINALS[w] for w in words if w in self._ORDINALS), None)
        fuzzy = re.compile(re.escape(selector), re.I)

        def youtube_results(frame: Any):
            domain = (urlsplit(frame.url).hostname or "").rstrip(".").lower()
            if domain == "youtube.com" or domain.endswith(".youtube.com"):
                return frame.locator("a#video-title")
            return None

        # Resolve equally precise strategies together across all frames.
        # Ordinal cues apply to descriptions and never change literal matches.
        groups = []
        if not by_text:
            groups.append(([("css", lambda frame: frame.locator(selector))], None))
        groups.extend(
            [
                (
                    [
                        (
                            "text-exact",
                            lambda frame: frame.get_by_text(selector, exact=True),
                        )
                    ],
                    None,
                ),
                ([("text-fuzzy", lambda frame: frame.get_by_text(fuzzy))], None),
            ]
        )
        if "video" in words or "result" in words:
            groups.append(([("youtube-result", youtube_results)], ordinal))
        keywords = [w for w in words if w not in self._DESC_STOPWORDS]
        if keywords:
            pattern = re.compile(".*".join(re.escape(word) for word in keywords), re.I)
            roles = [role for role in ("link", "button") if role in words]
            groups.append(
                (
                    [
                        (
                            f"role-{role}",
                            lambda frame, role=role: frame.get_by_role(
                                role, name=pattern
                            ),
                        )
                        for role in (roles or ["link", "button"])
                    ],
                    ordinal,
                )
            )

        frames = page.frames
        for lookups, group_ordinal in groups:
            handles = []
            candidates = []
            try:
                for frame_index, frame in enumerate(frames):
                    for strategy, lookup in lookups:
                        time_remaining_ms()
                        try:
                            locator = lookup(frame)
                            count = locator.count() if locator is not None else 0
                        except Exception:
                            continue  # For example, text can be invalid CSS.
                        if count == 0:
                            continue
                        if count > _MAX_CLICK_MATCHES:
                            raise RuntimeError(
                                "The selector matches too many elements. "
                                "Use a more precise selector."
                            )
                        # Pin this DOM node so removal cannot redirect the click.
                        snapshot = locator.element_handles()
                        handles.extend(snapshot)
                        if len(handles) > _MAX_CLICK_MATCHES:
                            raise RuntimeError(
                                "The selector matches too many elements. "
                                "Use a more precise selector."
                            )
                        for handle in snapshot:
                            time_remaining_ms()
                            if handle.is_visible() and handle.is_enabled():
                                candidates.append((handle, strategy, frame_index))
                if not candidates:
                    continue
                if group_ordinal is None and len(candidates) != 1:
                    raise RuntimeError(
                        "Multiple visible elements match this selector. "
                        "Use a more precise selector."
                    )
                chosen = 0 if group_ordinal is None else group_ordinal
                if chosen >= len(candidates):
                    raise RuntimeError(
                        "The requested position does not match a visible element."
                    )
                target, strategy, frame_index = candidates[chosen]
                timeout = min(8000 if frame_index == 0 else 4000, time_remaining_ms())
                try:
                    target.click(timeout=timeout)
                except Exception as exc:
                    # Playwright can fail after dispatch, e.g. if navigation
                    # started but did not finish. Never try another target.
                    raise _ClickOutcomeUnknown(
                        "The browser could not confirm this click. Inspect the "
                        "current page before deciding whether to retry."
                    ) from exc
                return strategy
            finally:
                for handle in handles:
                    try:
                        handle.dispose()
                    except Exception:
                        logger.debug("Could not release browser element", exc_info=True)

        hint = ""
        try:
            texts = page.evaluate(
                """(limit) => Array.from(
                    document.querySelectorAll('a, button, [role=button], [role=link]')
                ).map(e => (e.innerText || e.getAttribute('aria-label') || '').trim())
                 .filter(t => t && t.length < 80)
                 .filter((t, i, arr) => arr.indexOf(t) === i)
                 .slice(0, limit)""",
                15,
            )
            if texts:
                hint = " Visible clickable elements: " + "; ".join(texts)
        except Exception:
            pass
        raise RuntimeError(
            f"No element matched '{selector}' in the main page or its"
            f" {len(frames) - 1} iframe(s).{hint}"
        )


# ---------------------------------------------------------------------------
# Tool 3: BrowserTypeTool
# ---------------------------------------------------------------------------


@ToolRegistry.register("browser_type")
class BrowserTypeTool(BaseTool):
    """Type text into a form field."""

    tool_id = "browser_type"
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_type",
            description=(
                "Type text into a form field on the current page."
                " Can clear the field first or append to existing content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the input field.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type into the field.",
                    },
                    "clear": {
                        "type": "boolean",
                        "description": (
                            "If true, clear the field before typing. Default: true."
                        ),
                    },
                },
                "required": ["selector", "text"],
            },
            category="browser",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        selector = params.get("selector", "")
        text = params.get("text", "")

        if not selector:
            return ToolResult(
                tool_name="browser_type",
                content="No selector provided.",
                success=False,
            )
        if not text:
            return ToolResult(
                tool_name="browser_type",
                content="No text provided.",
                success=False,
            )

        clear = params.get("clear", True)

        try:

            def type_into(page: Any) -> None:
                if clear:
                    page.fill(selector, text)
                else:
                    page.type(selector, text)

            _run_browser_action(type_into)

            return ToolResult(
                tool_name="browser_type",
                content=f"Typed text into: {selector}",
                success=True,
                metadata={"selector": selector},
            )
        except ImportError:
            return ToolResult(
                tool_name="browser_type",
                content=(
                    "playwright not installed. Install with: uv sync --extra browser"
                ),
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_type",
                content=f"Type error: {exc}",
                success=False,
            )


# ---------------------------------------------------------------------------
# Tool 4: BrowserScreenshotTool
# ---------------------------------------------------------------------------


@ToolRegistry.register("browser_screenshot")
class BrowserScreenshotTool(BaseTool):
    """Take a screenshot of the current page."""

    tool_id = "browser_screenshot"
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_screenshot",
            description=(
                "Take a screenshot of the current browser page."
                " Returns the screenshot as base64-encoded data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional file path to save the screenshot.",
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": (
                            "If true, capture the full scrollable page. Default: false."
                        ),
                    },
                },
            },
            category="browser",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        path = params.get("path")
        full_page = params.get("full_page", False)

        try:
            screenshot_bytes = _run_browser_action(
                lambda page: page.screenshot(full_page=full_page)
            )

            if path:
                with open(path, "wb") as f:
                    f.write(screenshot_bytes)

            b64_data = base64.b64encode(screenshot_bytes).decode("utf-8")

            description = "Screenshot taken"
            if full_page:
                description += " (full page)"
            if path:
                description += f", saved to {path}"

            return ToolResult(
                tool_name="browser_screenshot",
                content=description,
                success=True,
                metadata={"screenshot_base64": b64_data},
            )
        except ImportError:
            return ToolResult(
                tool_name="browser_screenshot",
                content=(
                    "playwright not installed. Install with: uv sync --extra browser"
                ),
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_screenshot",
                content=f"Screenshot error: {exc}",
                success=False,
            )


# ---------------------------------------------------------------------------
# Tool 5: BrowserExtractTool
# ---------------------------------------------------------------------------


@ToolRegistry.register("browser_extract")
class BrowserExtractTool(BaseTool):
    """Extract content from the current page."""

    tool_id = "browser_extract"
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="browser_extract",
            description=(
                "Extract content from the current browser page."
                " Supports extracting text, links, or tables."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector to extract from. Default: 'body'."
                        ),
                    },
                    "extract_type": {
                        "type": "string",
                        "description": (
                            "Type of extraction: 'text', 'links',"
                            " or 'tables'. Default: 'text'."
                        ),
                    },
                },
            },
            category="browser",
        )

    def execute(self, **params: Any) -> ToolResult:
        selector = params.get("selector", "body")
        extract_type = params.get("extract_type", "text")

        if extract_type not in ("text", "links", "tables"):
            return ToolResult(
                tool_name="browser_extract",
                content=(
                    f"Invalid extract_type: '{extract_type}'."
                    " Must be 'text', 'links', or 'tables'."
                ),
                success=False,
            )

        try:

            def extract(page: Any) -> tuple[str, dict[str, Any]]:
                if extract_type == "text":
                    content = page.inner_text(selector)
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[Content truncated]"
                    return content, {
                        "selector": selector,
                        "extract_type": extract_type,
                    }

                if extract_type == "links":
                    links = page.eval_on_selector_all(
                        f"{selector} a[href]",
                        """elements => elements.map(el => ({
                        href: el.href,
                        text: el.innerText.trim()
                    }))""",
                    )
                    lines = [
                        f"- [{link.get('text', '')}]({link.get('href', '')})"
                        for link in links
                    ]
                    content = "\n".join(lines) if lines else "No links found."
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[Content truncated]"
                    return content, {
                        "selector": selector,
                        "extract_type": extract_type,
                        "num_links": len(links),
                    }

                tables_text = page.eval_on_selector_all(
                    f"{selector} table",
                    """elements => elements.map(el => el.innerText)""",
                )
                content = (
                    "\n\n---\n\n".join(tables_text)
                    if tables_text
                    else "No tables found."
                )
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[Content truncated]"
                return content, {
                    "selector": selector,
                    "extract_type": extract_type,
                    "num_tables": len(tables_text),
                }

            content, metadata = _run_browser_action(extract)
            return ToolResult(
                tool_name="browser_extract",
                content=content,
                success=True,
                metadata=metadata,
            )

        except ImportError:
            return ToolResult(
                tool_name="browser_extract",
                content=(
                    "playwright not installed. Install with: uv sync --extra browser"
                ),
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="browser_extract",
                content=f"Extract error: {exc}",
                success=False,
            )


__all__ = [
    "BrowserNavigateTool",
    "BrowserClickTool",
    "BrowserTypeTool",
    "BrowserScreenshotTool",
    "BrowserExtractTool",
]
