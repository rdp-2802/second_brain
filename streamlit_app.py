"""
Second Brain — Streamlit Frontend
==================================
Talks to the FastAPI backend at API_URL (default http://localhost:8000).
Run:
  uvicorn app.main:app --reload --port 8000  # backend
  streamlit run streamlit_app.py            # frontend (this file)

Notes
-----
* Auth is cookie-session based. A single requests.Session (kept in
  st.session_state) holds the session_id cookie — do NOT recreate it per call.
* Backend cookie is `secure=True`, but server-side `requests` resends it over
  plain HTTP locally. In production, serve the API over HTTPS.
* POST /chat/{id}/messages is slow (retrieval + Gemini + optional
  summarisation/ingestion). Timeout is set to 300s.
"""

import os
from datetime import datetime, timezone

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SHORT = 15
TIMEOUT_LONG = 300  # send-message is slow by design

st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def http() -> requests.Session:
    if "http" not in st.session_state:
        st.session_state.http = requests.Session()
    return st.session_state.http


def _url(path: str) -> str:
    return f"{API_URL}{path}"


def _handle_401(resp: requests.Response) -> bool:
    """If 401, clear auth state and force re-login. Returns True if handled."""
    if resp.status_code == 401:
        st.session_state.logged_in = False
        st.session_state.active_chat_id = None
        st.session_state.messages = []
        http().cookies.clear()
        return True
    return False


def _fmt_last_message_at(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # relative-ish
        delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return dt.strftime("%b %d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# API wrappers — auth
# ---------------------------------------------------------------------------


def api_signup(name: str, email: str, mobile: str, password: str):
    try:
        r = http().post(
            _url("/auth/signup"),
            json={"name": name, "email": email, "mobile": mobile, "password": password},
            timeout=TIMEOUT_SHORT,
        )
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the backend running?")
        return None
    if r.status_code == 200:
        return r
    detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
    st.error(f"Signup failed ({r.status_code}): {detail}")
    return None


def api_login(email: str | None, mobile: str | None, password: str):
    payload: dict = {"password": password}
    if email and email.strip():
        payload["email"] = email.strip()
    if mobile and mobile.strip():
        payload["mobile"] = mobile.strip()
    try:
        r = http().post(_url("/auth/login"), json=payload, timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the backend running?")
        return None
    if r.status_code == 200:
        return r
    try:
        detail = r.json().get("detail")
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
    except Exception:
        detail = r.text
    st.error(f"Login failed ({r.status_code}): {detail}")
    return None


def api_logout() -> bool:
    try:
        r = http().post(_url("/auth/logout"), timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return False
    if r.status_code == 200:
        return True
    if _handle_401(r):
        return True
    st.error(f"Logout failed ({r.status_code}): {r.text[:300]}")
    return False


def api_delete_account() -> bool:
    try:
        r = http().post(_url("/auth/delete_account"), timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return False
    if r.status_code == 200:
        return True
    if _handle_401(r):
        return False
    st.error(f"Delete account failed ({r.status_code}): {r.text[:300]}")
    return False


# ---------------------------------------------------------------------------
# API wrappers — chat
# ---------------------------------------------------------------------------


def api_list_chats():
    try:
        r = http().get(_url("/chat/"), timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    if r.status_code == 200:
        return r.json().get("chats", [])
    if _handle_401(r):
        st.warning("Session expired. Please log in again.")
        st.rerun()
    st.error(f"Could not load chats ({r.status_code}): {r.text[:300]}")
    return None


def api_create_chat(title: str | None):
    payload = {}
    if title and title.strip():
        payload["title"] = title.strip()
    try:
        r = http().post(_url("/chat/"), json=payload, timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    if r.status_code == 200:
        return r.json()
    if _handle_401(r):
        st.warning("Session expired. Please log in again.")
        st.rerun()
    st.error(f"Create chat failed ({r.status_code}): {r.text[:300]}")
    return None


def api_load_chat(chat_id: str):
    try:
        r = http().get(_url(f"/chat/{chat_id}"), timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        st.error("Chat not found (or not yours).")
        return None
    if _handle_401(r):
        st.warning("Session expired. Please log in again.")
        st.rerun()
    st.error(f"Could not load chat ({r.status_code}): {r.text[:300]}")
    return None


def api_rename_chat(chat_id: str, title: str):
    try:
        r = http().put(_url(f"/chat/{chat_id}"), json={"title": title}, timeout=TIMEOUT_SHORT)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        st.error("Chat not found.")
        return None
    if _handle_401(r):
        st.warning("Session expired.")
        st.rerun()
    try:
        detail = r.json().get("detail")
    except Exception:
        detail = r.text
    st.error(f"Rename failed ({r.status_code}): {detail}")
    return None


def api_delete_chat(chat_id: str) -> bool:
    try:
        r = http().delete(_url(f"/chat/{chat_id}"), timeout=TIMEOUT_LONG)
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return False
    if r.status_code == 200:
        return True
    if r.status_code == 404:
        st.error("Chat not found.")
        return False
    if _handle_401(r):
        st.warning("Session expired.")
        st.rerun()
    st.error(f"Delete failed ({r.status_code}): {r.text[:300]}")
    return False


def api_send_message(chat_id: str, content: str):
    """Returns dict {chat_id, title, messages:[assistant_msg]} or None."""
    try:
        r = http().post(
            _url(f"/chat/{chat_id}/messages"),
            json={"content": content},
            timeout=TIMEOUT_LONG,
        )
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}.")
        return None
    except requests.Timeout:
        st.error("The assistant is taking longer than expected (timeout). Your message was not saved — please retry.")
        return None
    if r.status_code == 200:
        return r.json()
    if r.status_code == 502:
        st.error("The assistant could not respond (LLM error). Your message was not saved — please try again.")
        return None
    if r.status_code == 404:
        st.error("Chat not found.")
        return None
    if _handle_401(r):
        st.warning("Session expired. Please log in again.")
        st.rerun()
    st.error(f"Send failed ({r.status_code}): {r.text[:500]}")
    return None


# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------
for k, v in {
    "logged_in": False,
    "chats": [],
    "active_chat_id": None,
    "messages": [],  # list[{order, role, content}]
    "active_chat_title": "New Chat",
    "sending": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def refresh_chats():
    chats = api_list_chats()
    if chats is not None:
        st.session_state.chats = chats


def select_chat(chat_id: str):
    data = api_load_chat(chat_id)
    if data is None:
        return
    st.session_state.active_chat_id = data["chat_id"]
    st.session_state.active_chat_title = data.get("title", "New Chat")
    # backend already sorts by order
    st.session_state.messages = data.get("messages", [])


# ---------------------------------------------------------------------------
# Auth view (when not logged in)
# ---------------------------------------------------------------------------
if not st.session_state.logged_in:
    st.title("🧠 Second Brain")
    st.caption(
        f"Your personal long-term memory assistant. Backend: `{API_URL}` "
        "(override with `API_URL` env var)."
    )

    try:
        ping = http().get(_url("/"), timeout=3)
        if ping.status_code != 200:
            st.warning(f"Backend at {API_URL} looks unhealthy (GET / → {ping.status_code}).")
    except Exception:
        st.warning(f"Backend at {API_URL} is not reachable yet. Start it with `uvicorn app.main:app --port 8000`.")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        st.subheader("Log in")
        st.caption("Provide **email or mobile** (at least one) plus password.")
        with st.form("login_form"):
            le = st.text_input("Email", placeholder="you@example.com")
            lm = st.text_input("Mobile", placeholder="10-digit mobile (as stored)")
            lp = st.text_input("Password", type="password")
            lb = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if lb:
            if not lp.strip():
                st.error("Password is required.")
            elif not (le.strip() or lm.strip()):
                st.error("Provide at least email or mobile.")
            else:
                r = api_login(le, lm, lp)
                if r is not None:
                    st.session_state.logged_in = True
                    refresh_chats()
                    # auto-select most recent if any
                    if st.session_state.chats:
                        select_chat(st.session_state.chats[0]["chat_id"])
                    st.success("Logged in.")
                    st.rerun()

    with tab_signup:
        st.subheader("Create an account")
        st.caption("All fields are required. Email and mobile must be unique.")
        with st.form("signup_form"):
            sn = st.text_input("Name", placeholder="Rishi")
            se = st.text_input("Email *", placeholder="you@example.com")
            sm = st.text_input("Mobile *", placeholder="10-digit number")
            sp = st.text_input("Password *", type="password")
            sb = st.form_submit_button("Sign up", type="primary", use_container_width=True)
        if sb:
            if not (sn.strip() and se.strip() and sm.strip() and sp.strip()):
                st.error("All fields are required.")
            else:
                r = api_signup(sn.strip(), se.strip(), sm.strip(), sp)
                if r is not None:
                    st.session_state.logged_in = True
                    refresh_chats()
                    st.success("Account created and logged in.")
                    st.rerun()

    st.divider()
    st.caption("Tip: run the backend and frontend in two terminals — `uvicorn` and `streamlit run`.")
    st.stop()

# ---------------------------------------------------------------------------
# Main app (when logged in)
# ---------------------------------------------------------------------------

# Sidebar — chat list + account actions
with st.sidebar:
    st.title("🧠 Second Brain")
    st.caption("Long-term memory chat")

    # New chat
    with st.expander("➕ New chat", expanded=False):
        new_title = st.text_input(
            "Title (optional)",
            placeholder="Leave blank for 'New Chat'",
            key="new_chat_title",
            max_chars=200,
        )
        if st.button("Create", type="primary", use_container_width=True, key="create_chat_btn"):
            data = api_create_chat(new_title)
            if data:
                refresh_chats()
                st.session_state.active_chat_id = data["chat_id"]
                st.session_state.active_chat_title = data.get("title", "New Chat")
                st.session_state.messages = data.get("messages", [])
                st.rerun()

    if st.button("↻ Refresh list", use_container_width=True):
        refresh_chats()
        if st.session_state.active_chat_id:
            select_chat(st.session_state.active_chat_id)
        st.rerun()

    st.divider()

    # Chat list
    if not st.session_state.chats:
        st.caption("No chats yet. Create one above.")
    else:
        st.caption(f"{len(st.session_state.chats)} chat(s)")
        for ch in st.session_state.chats:
            is_active = ch["chat_id"] == st.session_state.active_chat_id
            label = ch["title"] or "Untitled"
            sub = _fmt_last_message_at(ch.get("last_message_at"))
            btn_label = f"{'▸ ' if is_active else ''}{label}"
            if sub:
                btn_label += f"  ·  {sub}"
            if st.button(
                btn_label,
                key=f"open_{ch['chat_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                select_chat(ch["chat_id"])
                st.rerun()

    st.divider()

    # Account area
    st.subheader("Account")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Log out", use_container_width=True):
            if api_logout():
                st.session_state.logged_in = False
                st.session_state.active_chat_id = None
                st.session_state.messages = []
                st.session_state.chats = []
                http().cookies.clear()
                st.success("Logged out.")
                st.rerun()
    with c2:
        if st.button("Delete account", type="primary", use_container_width=True):
            st.session_state.confirm_delete_account = True

    if st.session_state.get("confirm_delete_account"):
        st.warning("This deletes your account **and all chats/memories**. This cannot be undone.")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("Yes, delete", type="primary", use_container_width=True):
                if api_delete_account():
                    st.session_state.logged_in = False
                    st.session_state.active_chat_id = None
                    st.session_state.messages = []
                    st.session_state.chats = []
                    http().cookies.clear()
                    st.session_state.confirm_delete_account = False
                    st.success("Account deleted.")
                    st.rerun()
        with dc2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete_account = False
                st.rerun()

    st.caption(f"API: `{API_URL}`")

# Main header
if st.session_state.active_chat_id is None:
    st.title("Welcome 👋")
    st.info("Create or pick a chat from the sidebar to start. Your assistant remembers across conversations via long-term memory retrieval.")
    # show a hint about 502 semantics
    st.caption(
        "When the assistant replies, memory retrieval + Gemini run on the server — replies can take 10–40s. "
        "If you see *LLM failed to respond*, your message was **not** saved — just retry."
    )
else:
    # Title row with inline rename/delete
    h1, h2, h3 = st.columns([6, 1.5, 1.5])
    with h1:
        st.subheader(st.session_state.active_chat_title)
        st.caption(f"`{st.session_state.active_chat_id}`")
    with h2:
        with st.popover("✏️ Rename", use_container_width=True):
            nt = st.text_input("New title", value=st.session_state.active_chat_title, key="rename_input", max_chars=200)
            if st.button("Save", type="primary", use_container_width=True, disabled=st.session_state.sending):
                if not nt.strip():
                    st.error("Title cannot be empty.")
                else:
                    res = api_rename_chat(st.session_state.active_chat_id, nt.strip())
                    if res:
                        st.session_state.active_chat_title = res["title"]
                        refresh_chats()
                        st.success("Renamed.")
                        st.rerun()
    with h3:
        with st.popover("🗑️ Delete", use_container_width=True):
            st.warning("Delete this chat? Remaining messages are ingested first so memory isn't lost, then everything is removed.")
            if st.button("Delete", type="primary", use_container_width=True, disabled=st.session_state.sending):
                if api_delete_chat(st.session_state.active_chat_id):
                    st.session_state.active_chat_id = None
                    st.session_state.messages = []
                    st.session_state.active_chat_title = "New Chat"
                    refresh_chats()
                    st.success("Chat deleted.")
                    st.rerun()

    st.divider()

    # Messages
    for m in st.session_state.messages:
        role = m.get("role", "user")
        # Streamlit expects "user"/"assistant"
        avatar_role = "assistant" if role == "assistant" else "user"
        with st.chat_message(avatar_role):
            st.markdown(m.get("content", ""))

    # Composer — disabled while waiting for the slow LLM turn
    prompt = st.chat_input("Ask anything…", disabled=st.session_state.sending)
    if prompt is not None:
        content = prompt.strip()
        if not content:
            st.warning("Message cannot be empty.")
        else:
            # Optimistic user bubble
            st.session_state.messages.append({"order": 0, "role": "user", "content": content})
            st.session_state.sending = True
            with st.chat_message("user"):
                st.markdown(content)
            # Call backend with a spinner — this is the slow path
            with st.status("Thinking… searching memories and asking Gemini (this can take a while)…", expanded=True) as status:
                st.write("Retrieving your long-term memories and calling the model. Please wait — **do not resend**.")
                res = api_send_message(st.session_state.active_chat_id, content)
                if res is None:
                    # Failure: rollback was already done server-side — remove optimistic bubble
                    st.session_state.messages.pop()  # remove the optimistic user message
                    status.update(label="Assistant could not respond", state="error")
                    st.error("LLM failed to respond. Your message was not saved — please try again.")
                else:
                    # Success: backend returns exactly the new assistant message
                    assistant_msgs = res.get("messages", [])
                    if assistant_msgs:
                        for am in assistant_msgs:
                            st.session_state.messages.append(am)
                            with st.chat_message("assistant"):
                                st.markdown(am.get("content", ""))
                    # Title may have changed externally; use returned title
                    if res.get("title"):
                        st.session_state.active_chat_title = res["title"]
                    refresh_chats()
                    status.update(label="Done", state="complete")
            st.session_state.sending = False
            st.rerun()
