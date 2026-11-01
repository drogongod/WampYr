"""Launch an independent Prism server; weights/binary read from H:, all logs/cache in this copy."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
EXE = Path(r'H:\models\LucyWest\llamacpp_prismml\build\bin\Release\llama-server.exe')
# 2026-10-29, swapped from Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf to Ornith-1.5-9B-uncensored
# (Jonathan: "shown to be more reliable, less hallucinations"). Concrete, measured reason to trust
# that call, not just taking it on faith: the 27B model burned its ENTIRE completion budget on
# reasoning_content TWICE the same night, both times returning empty final `content`
# (finish_reason="length") for an ordinary factual question -- a real reliability failure, not a
# one-off. Q8_0 chosen over the smaller Q4_K_M variant also on disk, since the explicit goal here is
# reliability/fidelity, not speed; -c 8192 matches Ornith's own Modelfile's num_ctx=8192, not just
# copied from the old model by accident. Alias kept as "vampire-lucywest" on purpose -- see
# REFERENCE_MODEL in brickcluster_engine.py and ensure_server()'s own health check below, which
# validates by ALIAS (the donor ROLE), not by which underlying .gguf backs it; changing the alias
# would require also updating brickcluster_engine.py in lockstep for no real benefit.
MODEL = Path(r'F:\LLM\Ornith-1.5-9B-uncensored-GGUF\Ornith-1.5-9B-uncensored.Q8_0.gguf')
URL = 'http://127.0.0.1:8102'

def ready():
    try:
        with urllib.request.urlopen(URL + '/health', timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def ensure_server():
    if os.environ.get('VAMPIRE_HOST'):
        print('Using explicitly configured host:', os.environ['VAMPIRE_HOST'])
        return None
    if ready():
        # Dedicated port, verify the exposed model identity before trusting it.
        with urllib.request.urlopen(URL + '/v1/models', timeout=5) as r:
            models = json.load(r).get('data', [])
        if not any('vampire-lucywest' == m.get('id') for m in models):
            raise RuntimeError('Port 8102 is occupied by a different model; no process was stopped.')
        return None
    if not EXE.is_file() or not MODEL.is_file():
        raise FileNotFoundError('Prism server or LucyWest model is missing; see README.md.')
    folder = ROOT / 'data' / 'server'
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / ('server_' + uuid.uuid4().hex + '.log')
    env = os.environ.copy()
    env['CUDA_CACHE_PATH'] = str(folder / 'cuda-cache')
    env['XDG_CACHE_HOME'] = str(folder / 'cache')
    args = [str(EXE), '-m', str(MODEL), '--host', '127.0.0.1', '--port', '8102',
        '--alias', 'vampire-lucywest', '-c', '8192', '-np', '1', '-ngl', '99', '--reasoning', 'auto']
    with log.open('xb') as f:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=f, stderr=f,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    info = folder / ('process_' + uuid.uuid4().hex + '.json')
    with info.open('x', encoding='utf-8') as f:
        json.dump({'pid': proc.pid, 'command': args, 'log': str(log), 'port': 8102}, f, indent=2)
    print('Starting independent LucyWest host. Log:', log, flush=True)
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError('Host exited; inspect ' + str(log))
        if ready():
            print('Host ready:', URL, flush=True)
            return proc
        time.sleep(1)
    raise RuntimeError('Host still loading; inspect ' + str(log) + '. It was not stopped.')

if __name__ == '__main__':
    ensure_server()
