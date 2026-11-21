"""Generate plan images with stable-diffusion.cpp + FLUX.2-klein-4B on CPU (stdlib only).

usage: python3 pipeline/images.py PLAN.json refs|scenes SHARD NSHARDS OUTDIR
Models download once into $MODELS_DIR. Character refs come from assets/characters/<ID>.png
when present, so a look you approve stays fixed across every future video.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

M = os.environ.get("MODELS_DIR", "models")
HF = "https://huggingface.co/"
FILES = {
    "sd.zip": "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1106-532d2c1/"
              "sd-master-532d2c1-bin-Linux-Ubuntu-26.04-x86_64.zip",
    "dit.gguf": HF + "leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf",
    "llm.gguf": HF + "unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    # BFL's small decoder: the full VAE decode took 225 s per image on a 4-vCPU runner
    "vae.safetensors": HF + "black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors",
    "up.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}


def fetch(name):
    p = os.path.join(M, name)
    if not os.path.exists(p):  # curl: retries + resume, so one dropped connection doesn't kill a 4 GB download
        subprocess.run(["curl", "-fsSL", "--retry", "6", "--retry-all-errors", "-C", "-", "-o", p + ".part", FILES[name]],
                       check=True)
        os.replace(p + ".part", p)


def sd_cli():
    d = os.path.join(M, "sd")
    if not os.path.isdir(d):
        zipfile.ZipFile(os.path.join(M, "sd.zip")).extractall(d)
    for root, _, files in os.walk(d):
        if "sd-cli" in files:
            exe = os.path.join(root, "sd-cli")
            os.chmod(exe, 0o755)
            return exe
    sys.exit("sd-cli not found in the stable-diffusion.cpp release zip")


def small_ref(path):
    """-r images cost tokens like pixels; a ~0.26 MP copy keeps identity at a fraction of the compute."""
    from PIL import Image  # python3-pil on the runner; only needed when a scene uses references
    out = path[:-4] + ".r.png"
    if not os.path.exists(out):
        im = Image.open(path).convert("RGB")
        im.thumbnail((448, 672))
        im.save(out)
    return out


def my_jobs(jobs, shard, n):
    """Longest-job-first onto the least-loaded shard; every shard computes the same split."""
    load, mine = [0.0] * n, []
    for j in sorted(jobs, key=lambda j: (-len(j["refs"]), j["name"])):
        k = load.index(min(load))
        load[k] += 1 + 0.45 * len(j["refs"])
        if k == shard:
            mine.append(j)
    return mine


def keyfile(path):
    return path[:-4] + ".key"


def page_job(exe, env, j, dst):
    """Real comic page: fetch, Real-ESRGAN 4x when the scan is small, cap at 4096 px so panel close-ups stay crisp."""
    from PIL import Image
    tmp = os.path.join(M, "pagetmp")
    os.makedirs(tmp, exist_ok=True)
    raw, src = os.path.join(tmp, "raw"), j["src"]
    if src.startswith(("http://", "https://")):
        ua = f"lore-engine/1.0 (+https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'lore-engine')})"
        subprocess.run(["curl", "-fsSL", "--retry", "4", "-A", ua, "-o", raw, src], check=True)
    else:
        shutil.copy(os.path.join("assets", "pages", src), raw)
    im = Image.open(raw).convert("RGB")  # PIL reads jpg/png/webp; sd-cli gets a clean png
    if max(im.size) < 2600:
        im.save(os.path.join(tmp, "in.png"))
        subprocess.run([exe, "-M", "upscale", "-i", os.path.join(tmp, "in.png"), "--upscale-model", f"{M}/up.pth",
                        "-o", os.path.join(tmp, "up.png"), "-t", str(os.cpu_count())], env=env, check=True)
        im = Image.open(os.path.join(tmp, "up.png")).convert("RGB")
    im.thumbnail((4096, 4096), Image.LANCZOS)
    im.save(dst)


def image_key(j, S, out):
    """Everything that changes the picture; an unchanged key means a cached image can be reused as-is."""
    if j.get("src"):
        local = os.path.join("assets", "pages", j["src"])
        ident = hashlib.sha1(open(local, "rb").read()).hexdigest() if os.path.exists(local) else j["src"]
        return hashlib.sha1(json.dumps([ident, "esrgan4x-4096", FILES["up.pth"]]).encode()).hexdigest()[:16]
    if j.get("asset") and os.path.exists(j["asset"]):
        return hashlib.sha1(open(j["asset"], "rb").read()).hexdigest()[:16]
    refs = [open(keyfile(os.path.join(out, r))).read() for r in j["refs"]]  # a new face re-draws its scenes
    blob = [j["prompt"], j["seed"], refs, S["image_width"], S["image_height"], S["image_steps"], S["upscale"],
            FILES["dit.gguf"], FILES["vae.safetensors"]]
    return hashlib.sha1(json.dumps(blob, default=str).encode()).hexdigest()[:16]


def main(plan_path, phase, shard, nshards, out):
    P = json.load(open(plan_path))
    S = P["settings"]
    cache = os.environ.get("IMAGE_CACHE", "")  # previous renders' images (actions/cache)
    os.makedirs(out, exist_ok=True)
    jobs = my_jobs(P["refs"] if phase == "refs" else P["images"], int(shard), int(nshards))
    todo = []
    for j in jobs:
        dst, key = os.path.join(out, j["name"]), image_key(j, S, out)
        hit = os.path.join(cache, j["name"]) if cache else ""
        if j.get("asset") and os.path.exists(j["asset"]):
            shutil.copy(j["asset"], dst)  # approved reference image wins over generation
        elif hit and os.path.exists(keyfile(hit)) and open(keyfile(hit)).read() == key:
            shutil.copy(hit, dst)
            print(f"cached {j['name']}")
        else:
            todo.append(j)
            continue
        open(keyfile(dst), "w").write(key)
    if not todo:
        return print("nothing to generate")
    os.makedirs(M, exist_ok=True)
    # refs feed FLUX.2 as -r inputs: keep them small (no 4x upscale) and portrait so the face gets the pixels
    upscale = phase != "refs" and str(S["upscale"]).lower() in ("yes", "true", "1")
    w, h = int(float(S["image_width"])), int(float(S["image_height"]))
    if phase == "refs":
        w, h = min(w, h), max(w, h)
    gen = any(not j.get("src") for j in todo)  # a shard holding only comic pages skips the 9 GB diffusion models
    need = ["sd.zip"] + (["dit.gguf", "llm.gguf", "vae.safetensors"] if gen else [])
    need += ["up.pth"] if (upscale and gen) or any(j.get("src") for j in todo) else []
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(fetch, need))
    exe = sd_cli()
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(exe))
    for i, j in enumerate(todo):
        dst = os.path.join(out, j["name"])
        if j.get("src"):
            t = time.time()
            page_job(exe, env, j, dst)
            open(keyfile(dst), "w").write(image_key(j, S, out))
            print(f"[{i + 1}/{len(todo)}] {j['name']} (comic page) {time.time() - t:.0f}s", flush=True)
            continue
        cmd = [exe, "--diffusion-model", f"{M}/dit.gguf", "--vae", f"{M}/vae.safetensors", "--llm", f"{M}/llm.gguf",
               "--cfg-scale", "1.0", "--sampling-method", "euler", "--steps", str(int(float(S["image_steps"]))),
               "-W", str(w), "-H", str(h),
               "-s", str(j["seed"]), "-p", j["prompt"], "-o", dst + ".tmp.png", "-t", str(os.cpu_count()), "--flash-attn", "on"]
        for r in j["refs"]:
            cmd += ["-r", small_ref(os.path.join(out, r))]
        if upscale:
            cmd += ["--upscale-model", f"{M}/up.pth"]
        t = time.time()
        for attempt in (1, 2):  # one retry for flaky OOM/IO; a second failure fails the job loudly
            if subprocess.run(cmd, env=env).returncode == 0:
                break
            if attempt == 2:
                sys.exit(f"sd-cli failed twice on {j['name']}")
        os.replace(dst + ".tmp.png", dst)
        open(keyfile(dst), "w").write(image_key(j, S, out))
        print(f"[{i + 1}/{len(todo)}] {j['name']} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
