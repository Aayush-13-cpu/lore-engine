"""Generate plan images with stable-diffusion.cpp + FLUX.2-klein-4B on CPU (stdlib only).

usage: python3 pipeline/images.py PLAN.json refs|scenes SHARD NSHARDS OUTDIR
Models download once into $MODELS_DIR. Character refs come from assets/characters/<ID>.png
when present, so a look you approve stays fixed across every future video.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

M = os.environ.get("MODELS_DIR", "models")
HF = "https://huggingface.co/"
FILES = {
    "sd.zip": "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1106-532d2c1/"
              "sd-master-532d2c1-bin-Linux-Ubuntu-26.04-x86_64.zip",
    "dit.gguf": HF + "leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf",
    "llm.gguf": HF + "unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    "vae.safetensors": HF + "black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors",
    "up.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}


def fetch(name):
    p = os.path.join(M, name)
    if not os.path.exists(p):
        urllib.request.urlretrieve(FILES[name], p + ".part")
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


def main(plan_path, phase, shard, nshards, out):
    P = json.load(open(plan_path))
    S = P["settings"]
    os.makedirs(out, exist_ok=True)
    jobs = (P["refs"] if phase == "refs" else P["images"])[int(shard)::int(nshards)]
    todo = []
    for j in jobs:
        dst = os.path.join(out, j["name"])
        if j.get("asset") and os.path.exists(j["asset"]):
            shutil.copy(j["asset"], dst)  # approved reference image wins over generation
        elif not os.path.exists(dst):
            todo.append(j)
    if not todo:
        return print("nothing to generate")
    os.makedirs(M, exist_ok=True)
    upscale = str(S["upscale"]).lower() in ("yes", "true", "1")
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(fetch, [f for f in FILES if f != "up.pth" or upscale]))
    exe = sd_cli()
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(exe))
    for i, j in enumerate(todo):
        dst = os.path.join(out, j["name"])
        cmd = [exe, "--diffusion-model", f"{M}/dit.gguf", "--vae", f"{M}/vae.safetensors", "--llm", f"{M}/llm.gguf",
               "--cfg-scale", "1.0", "--sampling-method", "euler", "--steps", str(int(float(S["image_steps"]))),
               "-W", str(int(float(S["image_width"]))), "-H", str(int(float(S["image_height"]))),
               "-s", str(j["seed"]), "-p", j["prompt"], "-o", dst + ".tmp.png", "-t", str(os.cpu_count()), "--flash-attn", "on"]
        for r in j["refs"]:
            cmd += ["-r", os.path.join(out, r)]
        if upscale:
            cmd += ["--upscale-model", f"{M}/up.pth"]
        t = time.time()
        for attempt in (1, 2):  # one retry for flaky OOM/IO; a second failure fails the job loudly
            if subprocess.run(cmd, env=env).returncode == 0:
                break
            if attempt == 2:
                sys.exit(f"sd-cli failed twice on {j['name']}")
        os.replace(dst + ".tmp.png", dst)
        print(f"[{i + 1}/{len(todo)}] {j['name']} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
