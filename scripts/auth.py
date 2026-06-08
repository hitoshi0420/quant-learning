"""
璁よ瘉涓庣敤鎴风鐞嗘ā鍧?- SQLite 鏁版嵁搴撳瓨鍌ㄧ敤鎴峰拰鑷€夎偂
- werkzeug 瀵嗙爜鍝堝笇
- 娉ㄥ唽/鐧诲綍/鐧诲嚭
- 鑷€夎偂澧炲垹鏌?- 绠＄悊鍛樼敤鎴?CRUD
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "auth.db"

_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """鑾峰彇鏁版嵁搴撹繛鎺?""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """鍒濆鍖栨暟鎹簱琛紝鎻掑叆榛樿绠＄悊鍛?""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock_code TEXT NOT NULL,
            added_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, stock_code)
        )
    """)
    conn.commit()

    # 鎻掑叆榛樿绠＄悊鍛橈紙濡傛灉涓嶅瓨鍦級
    import os, secrets
    existing = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?", ("admin",)
    ).fetchone()
    if not existing:
        admin_pw = os.environ.get("DASHBOARD_ADMIN_PASSWORD") or secrets.token_urlsafe(10)
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            ("admin", generate_password_hash(admin_pw)),
        )
        conn.commit()
        print(f"\n{'='*60}\n"
              f"  榛樿绠＄悊鍛樿处鍙峰凡鍒涘缓\n"
              f"  鐢ㄦ埛鍚? admin\n"
              f"  瀵嗙爜:   {admin_pw}\n"
              f"  璇风櫥褰曞悗灏藉揩淇敼瀵嗙爜\n"
              f"{'='*60}\n")
    elif check_password_hash(existing["password_hash"], os.environ.get("DASHBOARD_ADMIN_PASSWORD", "admin")):
        # 浠嶅湪浣跨敤鏃х‖缂栫爜瀵嗙爜锛岃嚜鍔ㄩ噸缃?        new_pw = os.environ.get("DASHBOARD_ADMIN_PASSWORD") or secrets.token_urlsafe(10)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_pw), existing["id"]),
        )
        conn.commit()
        print(f"\n{'='*60}\n"
              f"  [瀹夊叏] 妫€娴嬪埌 admin 浠嶄娇鐢ㄩ粯璁ゅ瘑鐮侊紝宸茶嚜鍔ㄩ噸缃甛n"
              f"  鏂板瘑鐮? {new_pw}\n"
              f"  璇峰Ε鍠勪繚绠★紒\n"
              f"{'='*60}\n")
    conn.close()


# ============================================================
# 璁よ瘉鍑芥暟
# ============================================================

def register_user(username: str, password: str, is_admin: bool = False) -> tuple:
    """娉ㄥ唽鐢ㄦ埛锛岃繑鍥?(success: bool, message: str)"""
    username = username.strip()
    if not username or not password:
        return False, "鐢ㄦ埛鍚嶅拰瀵嗙爜涓嶈兘涓虹┖"
    if len(username) < 2 or len(username) > 30:
        return False, "鐢ㄦ埛鍚嶉暱搴?2-30 瀛楃"
    if len(password) < 4:
        return False, "瀵嗙爜鑷冲皯 4 浣?

    with _db_lock:
        conn = _get_db()
        try:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                return False, "鐢ㄦ埛鍚嶅凡瀛樺湪"
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), 1 if is_admin else 0),
            )
            conn.commit()
            return True, "娉ㄥ唽鎴愬姛"
        except Exception as e:
            print(f"[auth] 娉ㄥ唽澶辫触: {e}")
            return False, "娉ㄥ唽澶辫触锛岃绋嶅悗閲嶈瘯"
        finally:
            conn.close()


def login_user(username: str, password: str) -> tuple:
    """楠岃瘉鐧诲綍锛岃繑鍥?(success: bool, user_dict | message: str)"""
    username = username.strip()
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                return False, "鐢ㄦ埛鍚嶆垨瀵嗙爜閿欒"
            if not check_password_hash(row["password_hash"], password):
                return False, "鐢ㄦ埛鍚嶆垨瀵嗙爜閿欒"
            return True, {
                "id": row["id"],
                "username": row["username"],
                "is_admin": bool(row["is_admin"]),
                "created_at": row["created_at"],
            }
        finally:
            conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """鏍规嵁 ID 鑾峰彇鐢ㄦ埛"""
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row:
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "is_admin": bool(row["is_admin"]),
                    "created_at": row["created_at"],
                }
            return None
        finally:
            conn.close()


def get_user_by_username(username: str) -> dict | None:
    """鏍规嵁鐢ㄦ埛鍚嶈幏鍙栫敤鎴?""
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "is_admin": bool(row["is_admin"]),
                    "created_at": row["created_at"],
                }
            return None
        finally:
            conn.close()


# ============================================================
# 鑷€夎偂鍑芥暟
# ============================================================

def get_watchlist(user_id: int) -> list[dict]:
    """鑾峰彇鐢ㄦ埛鑷€夎偂鍒楄〃锛堝惈鍩烘湰淇℃伅锛?""
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT stock_code, added_at FROM watchlist WHERE user_id = ? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
            return [{"code": r["stock_code"], "added_at": r["added_at"]} for r in rows]
        finally:
            conn.close()


def add_to_watchlist(user_id: int, stock_code: str) -> tuple:
    """娣诲姞鑷€夎偂"""
    stock_code = stock_code.strip()
    if not stock_code:
        return False, "鑲＄エ浠ｇ爜涓嶈兘涓虹┖"
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (user_id, stock_code) VALUES (?, ?)",
                (user_id, stock_code),
            )
            conn.commit()
            return True, "宸叉坊鍔?
        except Exception as e:
            print(f"[auth] 娣诲姞澶辫触: {e}")
            return False, "鎿嶄綔澶辫触锛岃绋嶅悗閲嶈瘯"
        finally:
            conn.close()


def remove_from_watchlist(user_id: int, stock_code: str) -> tuple:
    """鍒犻櫎鑷€夎偂"""
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND stock_code = ?",
                (user_id, stock_code),
            )
            conn.commit()
            return True, "宸插垹闄?
        except Exception as e:
            print(f"[auth] 鍒犻櫎澶辫触: {e}")
            return False, "鎿嶄綔澶辫触锛岃绋嶅悗閲嶈瘯"
        finally:
            conn.close()


# ============================================================
# 绠＄悊鍛樺嚱鏁?# ============================================================

def admin_list_users() -> list[dict]:
    """绠＄悊鍛橈細鍒楀嚭鎵€鏈夌敤鎴峰強鍏惰嚜閫夎偂鏁伴噺"""
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute("""
                SELECT u.id, u.username, u.is_admin, u.created_at,
                       COUNT(w.id) as wl_count
                FROM users u
                LEFT JOIN watchlist w ON u.id = w.user_id
                GROUP BY u.id
                ORDER BY u.id
            """).fetchall()
            return [
                {
                    "id": r["id"],
                    "username": r["username"],
                    "is_admin": bool(r["is_admin"]),
                    "created_at": r["created_at"],
                    "watchlist_count": r["wl_count"],
                }
                for r in rows
            ]
        finally:
            conn.close()


def admin_add_user(username: str, password: str, is_admin: bool = False) -> tuple:
    """绠＄悊鍛橈細娣诲姞鐢ㄦ埛"""
    return register_user(username, password, is_admin)


def admin_delete_user(admin_id: int, user_id: int) -> tuple:
    """绠＄悊鍛橈細鍒犻櫎鐢ㄦ埛锛堜笉鑳藉垹闄よ嚜宸憋級"""
    if admin_id == user_id:
        return False, "涓嶈兘鍒犻櫎鑷繁"
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return True, "宸插垹闄ょ敤鎴?
        except Exception as e:
            print(f"[auth] 鍒犻櫎澶辫触: {e}")
            return False, "鎿嶄綔澶辫触锛岃绋嶅悗閲嶈瘯"
        finally:
            conn.close()


def admin_get_user_watchlist(user_id: int) -> list[dict]:
    """绠＄悊鍛橈細鏌ョ湅鎸囧畾鐢ㄦ埛鐨勮嚜閫夎偂"""
    return get_watchlist(user_id)

