"""Isolated measurement lineages; cached assets are reused, records never mixed."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

from .core import POLICIES, PROTOCOL, alive, atomic_json, digest, exclusive, file_hash, read_json


def _source_inventory(root):
    paths = [root / 'bootstrap.py', root / 'requirements.txt', root / 'environment.lock.txt']
    paths += sorted((root / 'qspec_hybrid').glob('*.py'))
    paths += sorted((root / 'tests').glob('*.py'))
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def cached_environment(root, env):
    """Reconstruct the already-validated compiler paths, never install CUDA."""
    root = Path(root)
    receipt = read_json(root / 'results/cuda130_toolchain.json')
    if not receipt:
        return dict(env)
    from .toolchain import validated_environment
    return validated_environment(root, dict(env))


def _toolchain_inventory(root):
    if not (root / 'results/cuda130_toolchain.json').exists():
        return {}
    cached_environment(root, {})
    names = ['results/cuda130_toolchain.json', 'tooling/cuda130-pinned/nvidia/cu13/bin/nvcc',
             'tooling/cuda130-pinned/nvidia/cu13/include/cuda_runtime_api.h',
             'tooling/cuda130-pinned/nvidia/cu13/lib/libcudart.so.13']
    return {name:file_hash(root / name) for name in names if (root / name).is_file()}


def _saved_requests(root):
    path = root / 'results/requests.sqlite'
    if not path.exists():
        return {}
    with contextlib.closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('Historical request database failed integrity checking')
        return {key: digest(payload) for key, payload in conn.execute('SELECT key,payload FROM requests')}


def _checkpoint_receipts(root):
    result = {}
    folder = root / 'results/conversion_preflight'
    smoke = read_json(folder / 'vllm_smoke.json', {})
    if not (smoke.get('passed') and smoke.get('completed')):
        raise RuntimeError('Complete passing loading smoke evidence is required')
    expected = {(p, 0) for p in tuple(POLICIES)[1:]} | {('P1', 1), ('P7', 1)}
    evidence = smoke.get('results', [])
    if len(evidence) != len(expected) or not all(x.get('passed') for x in evidence) or {(x['policy'], x['k']) for x in evidence} != expected:
        raise RuntimeError('All nine expected loading smokes must have passed')
    for entry in evidence:
        if not entry.get('output_ids'):
            raise RuntimeError('Loading smoke has no generated output')
        if entry['k']:
            target = next(x for x in evidence if x['policy']==entry['policy'] and x['k']==0)
            if entry['output_ids'] != target['output_ids'] or not (entry.get('speculative') or {}).get('num_draft_tokens',0):
                raise RuntimeError('Native-MTP smoke lacks exact parity or draft proposals')
    for policy in tuple(POLICIES)[1:]:
        model = root / 'models' / policy
        report = read_json(folder / f'{policy}.json', {})
        manifest = read_json(model / 'export_manifest.json', {})
        content = manifest.get('weights_content_sha256')
        if (report.get('status') != 'pass' or not content or
                report.get('tensor_contract', {}).get('weights_content_sha256') != content):
            raise RuntimeError(f'Passing conversion evidence does not match cached weights: {policy}')
        for name, size in manifest.get('file_sizes', {}).items():
            path = model / name
            if Path(name).name != name or not path.is_file() or path.stat().st_size != size:
                raise RuntimeError(f'Cached checkpoint missing or changed: {policy}/{name}')
        if not manifest.get('file_sizes'):
            raise RuntimeError(f'Checkpoint has no file inventory: {policy}')
        for entry in (x for x in evidence if x['policy'] == policy):
            if entry.get('checkpoint_sha256') != content:
                raise RuntimeError(f'Loading evidence is not bound to cached weights: {policy}')
        result[policy] = {'weights_content_sha256': content,
                          'manifest_sha256': file_hash(model / 'export_manifest.json')}
    return result


def _revalidate_exports(child, expected):
    """CPU-only tensor/asset audit, using the existing worker and offline cache."""
    script = '''import json,sys
import torch
torch.set_num_threads(2)
from pathlib import Path
from qspec_hybrid.conversion import local_base,run_export_phase
from qspec_hybrid.core import read_json,atomic_json
from qspec_hybrid.audit import validate_runtime
root=Path(sys.argv[1])
expected=json.loads(sys.argv[2])
for phase in (("P1","P2","P3"),("P4","P5","P6","P7")):
    parent=Path(read_json(root/'lineage.json')['parent_root'])
    for policy in phase:
        atomic_json(parent/'results/fresh_lineage/status.json',{'state':'running','stage':'auditing '+policy,'candidate_root':str(root),'pilot_started':False,'full_scale_started':False})
        reports=run_export_phase(root,(policy,))
        assert len(reports)==1
        report=reports[0]
        assert report["status"]=="pass", report.get("blocking_reasons")
        assert report["tensor_contract"]["weights_content_sha256"]==expected[report["policy"]]["weights_content_sha256"]
for item in read_json(root/'results/conversion_preflight/vllm_smoke.json')['results']:
    validate_runtime(read_json(Path(item['server_folder'])/'runtime_precision.json'),item['policy'],item['k'])
print("Fresh lineage: all cached tensors, recipes and checkpoint assets verified offline.")
'''
    env = cached_environment(child, dict(os.environ, PYTHONPATH=str(child), PYTHONUNBUFFERED='1',
                             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    log = child / 'results/lineage_validation.log'
    with log.open('w') as output:
        subprocess.run([str(child / '.venv-hybrid/bin/python'), '-c', script,
                        str(child), json.dumps(expected)], cwd=child, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)


def active_root(parent):
    parent = Path(parent).resolve()
    pointer = read_json(parent / 'active_run.json')
    if not pointer:
        return parent
    child = (parent / pointer['relative_root']).resolve()
    if child.parent != (parent / 'runs').resolve():
        raise RuntimeError('Invalid active run path; historical project preserved')
    receipt = read_json(child / 'lineage.json', {})
    if receipt.get('state') != 'ready' or receipt.get('run_id') != pointer.get('run_id'):
        raise RuntimeError('Active run preparation is incomplete; inspect lineage.json')
    return child


def validate_lineage(root):
    root = Path(root).resolve()
    receipt = read_json(root / 'lineage.json')
    if not receipt:
        return
    if receipt.get('state') != 'ready':
        raise RuntimeError('Fresh lineage preparation has not completed')
    if receipt.get('source_sha256') != _source_inventory(root):
        raise RuntimeError('Fresh lineage source/dependency lock changed; do not mix measurements')
    if receipt.get('protocol') != digest(PROTOCOL):
        raise RuntimeError('Fresh lineage protocol changed')
    if receipt.get('toolchain_sha256', {}) != _toolchain_inventory(root):
        raise RuntimeError('Fresh lineage cached compiler/runtime changed')
    prepared = read_json(root / 'data/prepared.json')
    if not prepared or prepared.get('input_hash') != receipt['input_hash']:
        raise RuntimeError('Fresh lineage frozen input identity changed')
    if _checkpoint_receipts(root) != receipt['checkpoints']:
        raise RuntimeError('Fresh lineage cached checkpoint/evidence changed')


def prepare_fresh_lineage(parent):
    """Prepare and select a clean run, never launch pilot/full or alter parent data."""
    parent = Path(parent).resolve()
    if (parent/'lineage.json').exists():
        validate_lineage(parent)
        return parent
    with exclusive(parent):
        current = active_root(parent)
        for root in {parent, current}:
            for name in ('job.json', 'conversion_job.json'):
                if alive(read_json(root / name, {})):
                    raise RuntimeError('Stop the active worker before preparing another lineage')
        if current != parent:
            validate_lineage(current)
            return current  # Idempotent rerun: resume this lineage, do not reset it.
        for relative in ('.venv-hybrid/bin/python', '.environment.json', 'environment.lock.txt',
                         'data/prepared.json', 'revisions.json', 'tooling/modelopt/.git'):
            if not (parent / relative).exists():
                raise RuntimeError(f'Cached prerequisite missing: {relative}; no installation/download attempted')
        checkpoints = _checkpoint_receipts(parent)
        prepared = read_json(parent / 'data/prepared.json')
        if prepared.get('protocol') != digest(PROTOCOL) or prepared.get('input_hash') != digest(
                {k: v for k, v in prepared.items() if k != 'input_hash'}):
            raise RuntimeError('Frozen input manifest/protocol failed validation')
        before = _saved_requests(parent)
        source = _source_inventory(parent)
        toolchain = _toolchain_inventory(parent)
        run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
        child = parent / 'runs' / run_id
        child.mkdir(parents=True, exist_ok=False)
        history = child / 'results/history'
        history.mkdir(parents=True)
        receipt = {'schema': 1, 'state': 'preparing', 'run_id': run_id, 'parent_root': str(parent),
                   'created_at': time.time(), 'source_sha256': source,
                   'toolchain_sha256': toolchain,
                   'input_hash': prepared['input_hash'], 'protocol': digest(PROTOCOL),
                   'checkpoints': checkpoints, 'historical_request_count': len(before),
                   'historical_payload_hashes': before, 'full_scale_started': False,
                   'pilot_started': False, 'reuses_environment_and_weights': True}
        atomic_json(child / 'lineage.json', receipt)
        try:
            for relative in source:
                target = child / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parent / relative, target)
            for name in ('.environment.json', 'revisions.json', 'protocol.json', 'notebook_template.py', 'notebook.py', 'README.md'):
                if (parent / name).exists():
                    shutil.copy2(parent / name, child / name)
            for name in ('.venv-hybrid', 'tooling', 'sources'):
                if not (parent / name).is_dir():
                    raise RuntimeError(f'Cached resource missing: {name}')
                (child / name).symlink_to((parent / name).resolve(), target_is_directory=True)
            # Copy small inputs/manifests, hardlink immutable images, never the result DB.
            (child / 'data').mkdir()
            for path in sorted((parent / 'data').rglob('*')):
                if not path.is_file():
                    continue
                target = child / path.relative_to(parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                if 'images' in path.relative_to(parent / 'data').parts:
                    try:
                        os.link(path, target)
                    except OSError:
                        shutil.copy2(path, target)
                else:
                    shutil.copy2(path, target)
            for split in ('development', 'evaluation', 'heldout', 'calibration'):
                for case in prepared[split]:
                    image = child / case['image']
                    if not image.resolve().is_relative_to(child.resolve()) or file_hash(image) != case['image_file_sha256']:
                        raise RuntimeError(f'Frozen image missing, unsafe or changed: {case["id"]}')
            (child / 'models').mkdir()
            for policy in tuple(POLICIES)[1:]:
                (child / 'models' / policy).symlink_to((parent / 'models' / policy).resolve(), target_is_directory=True)
            audit_folder = child / 'results/conversion_preflight'
            audit_folder.mkdir()
            for path in (parent / 'results/conversion_preflight').glob('*.json'):
                if path.name not in ('status.json', 'completion_verification.json', 'notebook_delivery.json'):
                    shutil.copy2(path, audit_folder / path.name)
            smoke_archive=[]
            for entry in read_json(audit_folder/'vllm_smoke.json')['results']:
                source_folder=Path(entry['server_folder'])
                destination=history/'smoke_servers'/f'{entry["policy"]}-k{entry["k"]}'
                destination.mkdir(parents=True)
                for name in ('runtime_precision.json','launch.json','server.log'):
                    if (source_folder/name).is_file():
                        shutil.copy2(source_folder/name,destination/name)
                smoke_archive.append({'policy':entry['policy'],'k':entry['k'],
                                      'original':str(source_folder),'archived_relative':str(destination.relative_to(child))})
            atomic_json(history/'smoke_evidence.json',smoke_archive)
            # A closed SQLite backup is historical evidence only, never a resumable child DB.
            old_db = parent / 'results/requests.sqlite'
            if old_db.exists():
                with contextlib.closing(sqlite3.connect(old_db.as_uri() + '?mode=ro', uri=True)) as old, \
                     contextlib.closing(sqlite3.connect(history / 'historical_requests.sqlite')) as saved:
                    old.backup(saved)
                    saved.commit()
            if (parent / 'results/cuda130_toolchain.json').exists():
                shutil.copy2(parent / 'results/cuda130_toolchain.json', child / 'results/cuda130_toolchain.json')
            for name in ('environment.json', 'environment_mismatch.json', 'summary.json', 'summary.csv', 'comparisons.csv'):
                if (parent / 'results' / name).exists():
                    shutil.copy2(parent / 'results' / name, history / name)
            atomic_json(history / 'preservation.json', {'parent_root':str(parent),
                        'request_count':len(before), 'payload_hashes':before,
                        'included_in_new_measurements':False})
            _revalidate_exports(child, checkpoints)
            smoke = read_json(audit_folder / 'vllm_smoke.json')
            for policy in tuple(POLICIES)[1:]:
                path = audit_folder / f'{policy}.json'
                report = read_json(path)
                report['hardware_kernel_validation_completed'] = True
                report['hardware_kernel_validation'] = {
                    'passed':True, 'inherited_from':str(parent),
                    'checkpoint_sha256':checkpoints[policy]['weights_content_sha256'],
                    'server_evidence':[x['server_folder'] for x in smoke['results'] if x['policy']==policy],
                    'caveat':'Reused prior two-token eager smoke on identical exported tensor content; not a fresh pilot measurement.'}
                atomic_json(path, report)
            if _source_inventory(child) != source or _saved_requests(parent) != before:
                raise RuntimeError('Source or historical measurements changed during preparation')
            atomic_json(audit_folder / 'status.json', {'state':'passed',
                        'stage':'fresh lineage: cached exports revalidated; ready for pilot',
                        'conversion_gates_passed':True, 'pilot_resumed':False,
                        'full_scale_started':False, 'hardware_evidence_inherited_from':str(parent)})
            receipt['state'] = 'ready'
            atomic_json(child / 'lineage.json', receipt)
            atomic_json(child / 'status.json', {'state':'ready', 'mode':'pilot',
                        'stage':'Fresh pilot prepared; click Pilot test to measure', 'saved_records':0})
            # Publish only after every file/DB handle above is closed and validation passes.
            atomic_json(parent / 'active_run.json', {'run_id':run_id,
                        'relative_root':str(child.relative_to(parent)), 'prepared_at':time.time()})
            return child
        except BaseException as error:
            receipt.update(state='failed', failure=f'{type(error).__name__}: {error}')
            atomic_json(child / 'lineage.json', receipt)
            raise  # Partial preparation is retained, but never selected or launched.


def start_preparation(parent):
    """Detached preparation survives notebook HTTP/scratchpad timeouts."""
    import fcntl
    parent=Path(parent).resolve()
    with (parent/'lineage_launch.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        selected=active_root(parent)
        if selected!=parent or (parent/'lineage.json').exists():
            validate_lineage(selected)
            return {'state':'ready','run_root':str(selected)}
        old=read_json(parent/'lineage_job.json',{})
        if alive(old):
            return old
        for name in ('job.json','conversion_job.json'):
            if alive(read_json(parent/name,{})):
                raise RuntimeError('Stop the active worker before preparing a fresh pilot')
        python=parent/'.venv-hybrid/bin/python'
        if not python.is_file():
            raise RuntimeError('Cached worker Python is missing; no environment recreation attempted')
        folder=parent/'results/fresh_lineage'
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'status.json',{'state':'starting','stage':'preparing isolated pilot','pilot_started':False,'full_scale_started':False})
        env=cached_environment(parent,dict(os.environ,PYTHONPATH=str(parent),PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        log=folder/'worker.log'
        with log.open('ab') as output:
            process=subprocess.Popen([str(python),'-m','qspec_hybrid.lineage','--prepare-root',str(parent)],
                                     cwd=parent,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        from .core import process_identity
        job={'pid':process.pid,'birth':process_identity(process.pid),'mode':'prepare-fresh-pilot',
             'started':time.time(),'log':str(log)}
        atomic_json(parent/'lineage_job.json',job)
        return job


def main():
    import argparse
    import signal
    import traceback
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-root',required=True,type=Path)
    parent=parser.parse_args().prepare_root.resolve()
    status=parent/'results/fresh_lineage/status.json'
    def interrupted(*args):
        raise KeyboardInterrupt('Preparation stopped; historical measurements preserved')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        child=prepare_fresh_lineage(parent)
        validate_lineage(child)
        atomic_json(status,{'state':'ready','stage':'fresh pilot ready; click Pilot test / resume',
                    'run_root':str(child),'pilot_started':False,'full_scale_started':False})
        print('Fresh pilot prepared:',child,flush=True)
    except BaseException as error:
        atomic_json(status,{'state':'blocked','stage':'fresh pilot preparation failed',
                    'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc(),
                    'pilot_started':False,'full_scale_started':False})
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
