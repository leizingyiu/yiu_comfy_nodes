import os
import sys
import time
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from urllib.parse import urljoin
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ================= CONFIG =================

SYNC_INTERVAL = 3    # gap seconds
MAX_WORKERS = 6
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
HASH_CACHE_FILE = ".sync_hash_cache.json"

HTTP_USER = None   # e.g. "user"
HTTP_PASS = None   # e.g. "pass"

# ==========================================


# ---------- logging ----------

def setup_logger(target_dir, source, only_add):
    func = "setup_logger"
    parent = os.path.dirname(target_dir)
    name = os.path.basename(target_dir)

    log_dir = os.path.join(parent, f"{name}.sync-logs")
    os.makedirs(log_dir, exist_ok=True)

    start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = os.path.join(
        log_dir,
        f"{name}.synced.{start}~running.log"
    )

    handler = RotatingFileHandler(
        logfile,
        maxBytes=MAX_LOG_SIZE,
        backupCount=5,
        encoding="utf-8"
    )

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[handler],
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logging.info(f"[{func}] Sync started")
    logging.info(f"[{func}] Source: {source}")
    logging.info(f"[{func}] Target: {target_dir}")
    logging.info(f"[{func}] Mode: {'Add-only' if only_add else 'Full sync (with delete)'}")

    return logfile


def finalize_log(logfile):
    func = "finalize_log"
    end = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final = logfile.replace("~running", f"~{end}")
    try:
        os.rename(logfile, final)
        logging.info(f"[{func}] Log finalized: {final}")
    except Exception:
        logging.exception(f"[{func}] Failed to finalize log")


# ---------- utils ----------
def scan_local_tree(local_root):
    """
    扫描本地目录，返回 {相对路径: 绝对路径} 的映射
    """
    func = "scan_local_tree"
    logging.debug(f"[{func}] Scanning {local_root}")
    files = {}
    for root, _, filenames in os.walk(local_root):
        for name in filenames:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, local_root)
            # 统一使用 POSIX 风格的路径分隔符（避免 Windows \ 问题）
            rel_path = rel_path.replace(os.sep, "/")
            files[rel_path] = full_path
    return files
def md5_bytes(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def load_hash_cache(target_dir):
    func = "load_hash_cache"
    path = os.path.join(target_dir, HASH_CACHE_FILE)
    if not os.path.exists(path):
        logging.debug(f"[{func}] No cache file")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.exception(f"[{func}] Failed to load cache")
        return {}


def save_hash_cache(target_dir, cache):
    func = "save_hash_cache"
    path = os.path.join(target_dir, HASH_CACHE_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        logging.exception(f"[{func}] Failed to save cache")

def is_remote_directory(session, url):
    """
    判断一个 URL 指向的是目录还是文件
    """
    try:
        resp = session.get(url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        return "text/html" in content_type.lower()
    except Exception:
        return False

# ---------- HTTP listing ----------

class DirParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href and href not in ("../", "/"):
                self.links.append(href)


def fetch_remote_tree(session, base_url, sub=""):
    func = "fetch_remote_tree"
    url = urljoin(base_url + "/", sub)
    logging.debug(f"[{func}] Fetching {url}")

    resp = session.get(url)
    resp.raise_for_status()

    parser = DirParser()
    parser.feed(resp.text)

    files = {}

    for name in parser.links:
        name = name.lstrip("/")

        # 跳过无效链接
        if name.startswith("?") or name.startswith("#"):
            continue

        full_url = urljoin(url, name)

        # ⭐⭐⭐ 判断是不是目录
        if is_remote_directory(session, full_url):
            # 目录：只递归，不记录
            next_sub = sub + name.rstrip("/") + "/"
            files.update(fetch_remote_tree(session, base_url, next_sub))
        else:
            # 文件：记录最终路径
            rel_path = sub + name
            files[rel_path] = full_url

    return files


# ---------- diff ----------

def diff_files(remote_files, local_dir, hash_cache):
    func = "diff_files"
    to_download = {}
    to_delete = []

    for path, url in remote_files.items():
        local = os.path.join(local_dir, path)
        if not os.path.exists(local):
            to_download[path] = url
        else:
            if path in hash_cache:
                # assume unchanged, skip
                continue
            to_download[path] = url

    for root, _, files in os.walk(local_dir):
        for f in files:
            if f == HASH_CACHE_FILE:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir)
            if rel not in remote_files:
                to_delete.append(rel)

    logging.debug(
        f"[{func}] download={len(to_download)} delete={len(to_delete)}"
    )
    return to_download, to_delete


# ---------- download ----------
# ---------- copy ----------
# def copy_one(src_path, local_path):
#     func = "copy_one"
#     try:
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         with open(src_path, "rb") as src, open(local_path, "wb") as dst:
#             data = src.read()
#             dst.write(data)
#         return md5_bytes(data)
#     except Exception:
#         logging.exception(f"[{func}] Failed {local_path}")
#         raise


def copy_one(src_path, local_path):
    func = "copy_one"
    try:
        # ⭐⭐⭐ 关键修复：确保父路径是目录，而不是文件
        parent = os.path.dirname(local_path)
        if os.path.exists(parent) and os.path.isfile(parent):
            logging.warning(
                f"[{func}] Parent path is file, removing: {parent}"
            )
            os.remove(parent)

        os.makedirs(parent, exist_ok=True)

        with open(src_path, "rb") as src, open(local_path, "wb") as dst:
            data = src.read()
            dst.write(data)

        return md5_bytes(data)

    except Exception:
        logging.exception(f"[{func}] Failed {local_path}")
        raise



def parallel_copy(target_dir, tasks, hash_cache):
    func = "parallel_copy"
    logging.info(f"[{func}] Start {len(tasks)} copies")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for path, src_path in tasks.items():
            safe_path = path.lstrip("/\\")
            local = os.path.join(target_dir, safe_path)
            futures[
                pool.submit(copy_one, src_path, local)
            ] = path
        for f in as_completed(futures):
            path = futures[f]
            try:
                h = f.result()
                hash_cache[path] = h
                logging.info(f"[{func}] OK {path}")
            except Exception:
                logging.error(f"[{func}] ERROR {path}")
# def download_one(session, url, local_path):
#     func = "download_one"
#     try:
#         resp = session.get(url)
#         resp.raise_for_status()
#         data = resp.content

#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         with open(local_path, "wb") as f:
#             f.write(data)

#         return md5_bytes(data)
#     except Exception:
#         logging.exception(f"[{func}] Failed {local_path}")
#         raise

def download_one(session, url, local_path):
    func = "download_one"
    try:
        # ⭐⭐⭐ 关键修复：确保父路径是目录，而不是文件
        parent = os.path.dirname(local_path)
        if os.path.exists(parent) and os.path.isfile(parent):
            logging.warning(
                f"[{func}] Parent path is file, removing: {parent}"
            )
            os.remove(parent)

        os.makedirs(parent, exist_ok=True)

        resp = session.get(url)
        resp.raise_for_status()
        data = resp.content

        with open(local_path, "wb") as f:
            f.write(data)

        return md5_bytes(data)

    except Exception:
        logging.exception(f"[{func}] Failed {local_path}")
        raise



def parallel_download(session, target_dir, tasks, hash_cache):
    func = "parallel_download"
    logging.info(f"[{func}] Start {len(tasks)} downloads")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}

        for path, url in tasks.items():
            safe_path = path.lstrip("/\\")
            local = os.path.join(target_dir, safe_path)
            futures[
                pool.submit(download_one, session, url, local)
            ] = path

        for f in as_completed(futures):
            path = futures[f]
            try:
                h = f.result()
                hash_cache[path] = h
                logging.info(f"[{func}] OK {path}")
            except Exception:
                logging.error(f"[{func}] ERROR {path}")


# ---------- delete ----------

def delete_removed_files(target_dir, paths):
    func = "delete_removed_files"
    for rel in paths:
        full = os.path.join(target_dir, rel)
        try:
            os.remove(full)
            logging.info(f"[{func}] DELETE {rel}")
        except Exception:
            logging.exception(f"[{func}] Failed delete {rel}")


# ---------- main ----------
def prompt_for_args():
    """交互式获取参数"""
    print("No command-line arguments provided. Entering interactive mode.")
    source = input("Enter source (URL or local dir): ").strip()
    target = input("Enter target directory: ").strip()
    add_mode = input("Enable 'add-only' mode? (y/N): ").strip().lower()
    only_add = add_mode in ("y", "yes")
    return source, target, only_add


def parse_or_prompt_args():
    """解析命令行参数，若无则进入交互模式"""
    if len(sys.argv) == 1:
        # 无任何参数 → 交互
        return prompt_for_args()
    elif len(sys.argv) in (3, 4):
        source = sys.argv[1]
        target = sys.argv[2]
        if len(sys.argv) == 4:
            if sys.argv[3] == "-add":
                only_add = True
            else:
                print(f"Unknown option: {sys.argv[3]}")
                sys.exit(1)
        else:
            only_add = False
        return source, target, only_add
    else:
        print("Usage (CLI):   python script.py <source> <target> [-add]")
        print("Usage (Prompt): python script.py")
        sys.exit(1)

def main():
    source, target, only_add = parse_or_prompt_args()

    os.makedirs(target, exist_ok=True)

    logfile = setup_logger(target, source, only_add)

    session = requests.Session()
    if HTTP_USER and HTTP_PASS:
        session.auth = (HTTP_USER, HTTP_PASS)

    try:        
        while True:
            logging.debug("[main] Sync loop start")

            # 判断 source 是 HTTP 还是本地目录
            if source.lower().startswith(("http://", "https://")):
                remote_files = fetch_remote_tree(session, source)
                is_local_source = False
            else:
                # 假设是本地目录
                if not os.path.isdir(source):
                    logging.error("[main] Local source is not a directory")
                    sys.exit(1)
                remote_files = scan_local_tree(source)
                is_local_source = True

            hash_cache = load_hash_cache(target)
            to_download, to_delete = diff_files(remote_files, target, hash_cache)

            if to_download:
                if is_local_source:
                    parallel_copy(target, to_download, hash_cache)
                else:
                    parallel_download(session, target, to_download, hash_cache)

            if not only_add and to_delete:
                delete_removed_files(target, to_delete)

            save_hash_cache(target, hash_cache)
            logging.debug("[main] Sync loop end")
            time.sleep(SYNC_INTERVAL)

    except KeyboardInterrupt:
        logging.info("[main] Interrupted by user")
    except Exception:
        logging.exception("[main] Fatal error")
    finally:
        finalize_log(logfile)


if __name__ == "__main__":
    main()
