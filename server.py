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
MODEL = Path(r'H:\models\LucyWest\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf')
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
