"""Comic sources -> page images (stdlib only, so the image runners can use it too).

A page reference (the part before #panel / @box) is one of:
  FF48:5        page 5 of issue FF48 from the Sources sheet (cbz / zip / cbr / rar / pdf / folder of images)
  file.jpg      a single page in private/pages/ (your private comics repo) or assets/pages/ (public-domain only)
  https://...   a single page or a whole issue file on the web
Copyrighted issues live in your private repo, which the runners check out to private/ - never in this public repo.
"""
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
import zlib

ISSUE = re.compile(r"^([A-Za-z][\w-]*):(\d+)$")
IMG = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
ROOTS = ("private", "private/pages", "assets/sources", "assets/pages")


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def locate(loc):
    """Local path for a file name / relative path, searching the private repo first."""
    for root in ROOTS:
        p = os.path.join(root, loc)
        if os.path.exists(p):
            return p
    return None


def download(url, work):
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    path = os.path.join(work, f"dl_{zlib.crc32(url.encode()):08x}{ext}")
    if not os.path.exists(path):
        ua = f"lore-engine/1.0 (+https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'lore-engine')})"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=180) as r:
                data = r.read()
        except Exception as e:  # plain RuntimeError: survives worker pools, tells the user what to do
            raise RuntimeError(f"could not download {url}: {e} (put the file in your private repo instead)") from None
        with open(f"{path}.{os.getpid()}", "wb") as f:
            f.write(data)
        os.replace(f"{path}.{os.getpid()}", path)
    return path


def issue_pages(path, work):
    """All page images of an issue file or folder, in reading order (extracted once into work/)."""
    if os.path.isdir(path):
        d = path
    else:
        d = os.path.join(work, f"issue_{zlib.crc32(os.path.abspath(path).encode()):08x}")
        if not os.path.isdir(d):
            tmp = f"{d}.{os.getpid()}"
            os.makedirs(tmp, exist_ok=True)
            low = path.lower()
            if low.endswith((".cbz", ".zip")):
                zipfile.ZipFile(path).extractall(tmp)  # extractall drops absolute paths and '..'
            elif low.endswith((".cbr", ".rar", ".cb7", ".7z")):
                subprocess.run(["7z", "x", "-y", f"-o{tmp}", path], check=True, stdout=subprocess.DEVNULL)
            elif low.endswith(".pdf"):
                subprocess.run(["pdftoppm", "-r", "150", "-png", path, os.path.join(tmp, "p")], check=True)
            else:
                raise RuntimeError(f"{path}: unsupported comic file (use cbz, cbr, pdf or a folder of images)")
            try:
                os.replace(tmp, d)
            except OSError:  # another process finished first
                shutil.rmtree(tmp, ignore_errors=True)
    pages = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
             if f.lower().endswith(IMG) and "__MACOSX" not in r]
    return sorted(pages, key=lambda p: natural(os.path.relpath(p, d)))


def issue_file(sid, sources, work):
    loc = str(sources[sid]).strip()
    path = download(loc, work) if loc.startswith(("http://", "https://")) else locate(loc)
    if not path:
        raise RuntimeError(f"source {sid}: {loc} not found in private/ or assets/sources/ (is ASSETS_REPO set up?)")
    return path


def page_image(src, sources, work):
    """Image path for a page reference (see module doc)."""
    os.makedirs(work, exist_ok=True)
    m = ISSUE.match(src)
    if m and m.group(1) in sources:
        pages = issue_pages(issue_file(m.group(1), sources, work), work)
        n = int(m.group(2))
        if not 1 <= n <= len(pages):
            raise RuntimeError(f"{src}: issue {m.group(1)} has {len(pages)} pages")
        return pages[n - 1]
    if src in sources:  # a Sources row that is a single page image
        src = str(sources[src]).strip()
    if src.startswith(("http://", "https://")):
        return download(src, work)
    path = locate(src)
    if not path:
        raise RuntimeError(f"page {src} not found in private/pages/ or assets/pages/ (or add it to the Sources sheet)")
    return path


def _selfcheck(tmp):
    os.makedirs(tmp, exist_ok=True)
    cbz = os.path.join(tmp, "t.cbz")
    with zipfile.ZipFile(cbz, "w") as z:  # page10 must sort after page2
        for name in ("page10.png", "page2.png", "page1.png", "__MACOSX/page3.png"):
            z.writestr(name, name)
    got = [os.path.basename(p) for p in issue_pages(cbz, tmp)]
    assert got == ["page1.png", "page2.png", "page10.png"], got
    assert os.path.basename(page_image("T:3", {"T": cbz}, tmp)) == "page10.png"
