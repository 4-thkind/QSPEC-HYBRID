import contextlib
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from qspec_hybrid import core, lineage


@pytest.fixture
def parent(tmp_path, monkeypatch):
    root=tmp_path/'project'
    root.mkdir()
    for name in ['bootstrap.py','requirements.txt','environment.lock.txt','qspec_hybrid/demo.py','tests/test_demo.py','.venv-hybrid/bin/python','tooling/modelopt/.git/HEAD','sources/evaluator/README']:
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('cached content')
    core.atomic_json(root/'.environment.json',{'identity':'existing-setup'})
    core.atomic_json(root/'revisions.json',{'model':{'revision':'pinned'}})
    image=root/'data/images/image.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'immutable-image')
    case={'id':'example','image':'data/images/image.png','image_file_sha256':core.file_hash(image)}
    prepared={'protocol':core.digest(core.PROTOCOL),'development':[case], 'evaluation':[], 'heldout':[], 'calibration':[]}
    prepared['input_hash']=core.digest(prepared)
    core.atomic_json(root/'data/prepared.json',prepared)
    results=root/'results'
    with core.Store(results) as store:
        store.put('old-record',{'identity':'historical','text':'preserve me'})
    core.atomic_json(results/'environment.json',{'identity':'historical'})
    core.atomic_json(results/'summary.json',{'saved_records':1})
    modes=[]
    for policy in tuple(core.POLICIES)[1:]:
        model=root/'models'/policy
        model.mkdir(parents=True)
        (model/'weights.safetensors').write_bytes(b'cached-weights')
        content='verified-'+policy
        core.atomic_json(model/'export_manifest.json',{'weights_content_sha256':content,'file_sizes':{'weights.safetensors':14}})
        core.atomic_json(results/'conversion_preflight'/f'{policy}.json',{'policy':policy,'status':'pass','tensor_contract':{'weights_content_sha256':content}})
        for k in ((0,1) if policy in ('P1','P7') else (0,)):
            modes.append({'policy':policy,'k':k,'passed':True,'checkpoint_sha256':content,'output_ids':[1,2],
                          'server_folder':str(results/'servers'/f'{policy}-{k}'),
                          'speculative':{'num_draft_tokens':1} if k else None})
    core.atomic_json(results/'conversion_preflight/vllm_smoke.json',{'passed':True,'completed':True,'results':modes})
    monkeypatch.setattr(lineage,'_revalidate_exports',lambda child,expected: None)
    return root


def test_prepare_fresh_lineage_never_mixes_or_rewrites_records(parent):
    before=lineage._saved_requests(parent)
    old_identity=(parent/'results/environment.json').read_bytes()
    child=lineage.prepare_fresh_lineage(parent)
    assert child.parent==parent/'runs'
    assert lineage.active_root(parent)==child
    assert lineage._saved_requests(parent)==before
    assert (parent/'results/environment.json').read_bytes()==old_identity
    assert not (child/'results/requests.sqlite').exists()
    assert not (child/'results/environment.json').exists()
    assert not (child/'pilot_gate.json').exists()
    assert not (child/'job.json').exists()
    receipt=core.read_json(child/'lineage.json')
    assert receipt['state']=='ready'
    assert receipt['historical_request_count']==1
    with contextlib.closing(sqlite3.connect(child/'results/history/historical_requests.sqlite')) as conn:
        assert conn.execute('SELECT key FROM requests').fetchall()==[('old-record',)]
    lineage.validate_lineage(child)


def test_preparation_reuses_environment_weights_and_input_bytes(parent):
    (parent/'notebook_template.py').write_text('# preserved notebook scaffold')
    (parent/'notebook.py').write_text('# current end-to-end notebook')
    child=lineage.prepare_fresh_lineage(parent)
    assert (child/'notebook_template.py').read_bytes()==(parent/'notebook_template.py').read_bytes()
    assert (child/'notebook.py').read_bytes()==(parent/'notebook.py').read_bytes()
    assert (child/'.venv-hybrid').is_symlink()
    assert (child/'.venv-hybrid').resolve()==parent/'.venv-hybrid'
    for policy in tuple(core.POLICIES)[1:]:
        assert (child/'models'/policy).resolve()==parent/'models'/policy
    assert (child/'data/prepared.json').read_bytes()==(parent/'data/prepared.json').read_bytes()
    assert (child/'data/images/image.png').stat().st_ino==(parent/'data/images/image.png').stat().st_ino


def test_prepare_is_idempotent_and_resumes_the_new_lineage(parent):
    child=lineage.prepare_fresh_lineage(parent)
    with core.Store(child/'results') as store:
        store.put('new-record',{'identity':'new-run'})
    assert lineage.prepare_fresh_lineage(parent)==child
    assert len(list((parent/'runs').iterdir()))==1
    assert len(lineage._saved_requests(child))==1
    assert len(lineage._saved_requests(parent))==1


@pytest.mark.parametrize('relative',['.environment.json','environment.lock.txt','data/prepared.json','.venv-hybrid/bin/python'])
def test_missing_cached_prerequisite_does_not_install_or_publish(parent,relative):
    (parent/relative).unlink()
    with pytest.raises(RuntimeError,match='Cached prerequisite'):
        lineage.prepare_fresh_lineage(parent)
    assert not (parent/'active_run.json').exists()


def test_failed_audit_keeps_history_and_does_not_publish(parent,monkeypatch):
    def fail(*args):
        raise RuntimeError('real protected tensor change')
    monkeypatch.setattr(lineage,'_revalidate_exports',fail)
    before=lineage._saved_requests(parent)
    with pytest.raises(RuntimeError,match='protected tensor'):
        lineage.prepare_fresh_lineage(parent)
    assert not (parent/'active_run.json').exists()
    assert lineage._saved_requests(parent)==before
    child=next((parent/'runs').iterdir())
    assert core.read_json(child/'lineage.json')['state']=='failed'


def test_bad_checkpoint_or_unbound_smoke_is_rejected(parent):
    path=parent/'results/conversion_preflight/vllm_smoke.json'
    smoke=core.read_json(path)
    smoke['results'][0]['checkpoint_sha256']='different'
    core.atomic_json(path,smoke)
    with pytest.raises(RuntimeError,match='not bound'):
        lineage.prepare_fresh_lineage(parent)
    assert not (parent/'active_run.json').exists()


def test_source_change_after_preparation_cannot_be_resumed(parent):
    child=lineage.prepare_fresh_lineage(parent)
    (child/'qspec_hybrid/demo.py').write_text('different code')
    with pytest.raises(RuntimeError,match='source/dependency lock changed'):
        lineage.validate_lineage(child)
    with pytest.raises(RuntimeError,match='source/dependency lock changed'):
        core.launch(child,'pilot')


def test_invalid_active_run_path_is_rejected(parent):
    core.atomic_json(parent/'active_run.json',{'relative_root':'../outside','run_id':'x'})
    with pytest.raises(RuntimeError,match='Invalid active run path'):
        lineage.active_root(parent)


def test_running_worker_prevents_preparation(parent,monkeypatch):
    core.atomic_json(parent/'job.json',{'pid':123,'birth':'known'})
    monkeypatch.setattr(lineage,'alive',lambda job:bool(job))
    with pytest.raises(RuntimeError,match='Stop the active worker'):
        lineage.prepare_fresh_lineage(parent)


def test_missing_smoke_and_mtp_parity_are_fail_closed(parent):
    path=parent/'results/conversion_preflight/vllm_smoke.json'
    smoke=core.read_json(path)
    smoke['results'][1]['output_ids']=[999]
    core.atomic_json(path,smoke)
    with pytest.raises(RuntimeError,match='exact parity'):
        lineage.prepare_fresh_lineage(parent)


def test_fresh_launch_is_offline_and_never_auto_promotes_to_full(parent,monkeypatch):
    child=lineage.prepare_fresh_lineage(parent)
    captured={}
    def popen(args,**kwargs):
        captured.update(args=args,**kwargs)
        return type('Process',(),{'pid':987654321})()
    monkeypatch.setattr(subprocess,'Popen',popen)
    core.launch(child,'pilot')
    assert captured['args'][-1]=='pilot'
    assert captured['env']['HF_HUB_OFFLINE']=='1'
    assert captured['env']['TRANSFORMERS_OFFLINE']=='1'


def test_child_bundle_contains_closed_historical_snapshot_and_live_notebook(parent):
    import zipfile
    child=lineage.prepare_fresh_lineage(parent)
    (parent.parent/'notebook.py').write_text('def _qrun_full_conversion_preflight(root, control):\n    pass\n')
    archive=core.bundle(child)
    with zipfile.ZipFile(archive) as stored:
        assert 'lineage.json' in stored.namelist()
        assert 'results/history/historical_requests.sqlite' in stored.namelist()
        assert 'data/images/image.png' in stored.namelist()
        assert 'live_notebook.py' in stored.namelist()
        assert stored.testzip() is None


def test_full_launch_fails_before_starting_any_worker(parent,monkeypatch):
    child=lineage.prepare_fresh_lineage(parent)
    def never(*args,**kwargs):
        pytest.fail('Full worker must not be launched without a pilot gate')
    monkeypatch.setattr(subprocess,'Popen',never)
    with pytest.raises(RuntimeError,match='requires a passing pilot'):
        core.launch(child,'full')


def _fake_cached_toolchain(parent, monkeypatch, header_version='13000'):
    from qspec_hybrid import toolchain
    cuda=parent/'tooling/cuda130-pinned/nvidia/cu13'
    (cuda/'bin').mkdir(parents=True)
    (cuda/'include').mkdir()
    (cuda/'lib').mkdir()
    (cuda/'bin/nvcc').write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    (cuda/'bin/nvcc').chmod(0o755)
    (cuda/'include/cuda_runtime_api.h').write_text(f'#define CUDART_VERSION {header_version}\n')
    (cuda/'lib/libcudart.so.13').write_bytes(b'pinned runtime')
    package=parent/'fake-site/nvidia/cu13'
    (package/'include').mkdir(parents=True)
    (package/'include/cublasLt.h').write_text('/* worker header */')
    (package/'lib').mkdir()
    for name in ('libcublas.so.13','libcublasLt.so.13'):
        (package/'lib'/name).write_bytes(b'worker library')
    monkeypatch.setattr(toolchain,'_worker_site',lambda worker: parent/'fake-site')
    core.atomic_json(parent/'results/cuda130_toolchain.json',{
        'schema':2,
        'sampler_preflight_passed':True,
        'file_sha256':{
            str(path.relative_to(parent)):core.file_hash(path)
            for path in (cuda/'bin/nvcc',cuda/'include/cuda_runtime_api.h',cuda/'lib/libcudart.so.13')
        },
    })
    return cuda


def test_cached_toolchain_paths_are_durable_and_never_install(parent,monkeypatch):
    cuda=_fake_cached_toolchain(parent,monkeypatch)
    env=lineage.cached_environment(parent,{'PATH':'/usr/bin'})
    assert env['FLASHINFER_NVCC']==str(cuda/'bin/nvcc')
    assert env['CUDA_HOME']==str(cuda)
    child=lineage.prepare_fresh_lineage(parent)
    lineage.validate_lineage(child)
    (cuda/'bin/nvcc').write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n# changed\n")
    with pytest.raises(RuntimeError,match='Pinned CUDA toolchain changed'):
        lineage.validate_lineage(child)


def test_wrong_cached_cuda_headers_fail_without_reinstallation(parent,monkeypatch):
    _fake_cached_toolchain(parent,monkeypatch,header_version='13030')
    with pytest.raises(RuntimeError,match='CUDA 13.0 headers'):
        lineage.cached_environment(parent,{})


def test_bundle_waits_for_exclusive_worker_lock_release(parent):
    with core.exclusive(parent):
        with pytest.raises(RuntimeError,match='experiment lock is released'):
            core.bundle(parent)
    assert core.bundle(parent).exists()


def test_bundle_refuses_another_active_worker(parent,monkeypatch):
    core.atomic_json(parent/'job.json',{'pid':123,'birth':'known'})
    monkeypatch.setattr(core,'alive',lambda job:bool(job))
    with pytest.raises(RuntimeError,match='Stop the active worker'):
        core.bundle(parent)


def test_background_preparation_is_detached_offline_and_deduplicated(parent,monkeypatch):
    captured=[]
    def popen(args,**kwargs):
        captured.append((args,kwargs))
        return type('Process',(),{'pid':987654321})()
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(core,'process_identity',lambda pid:'same-birth')
    first=lineage.start_preparation(parent)
    second=lineage.start_preparation(parent)
    assert first==second and len(captured)==1
    args,kwargs=captured[0]
    assert kwargs['start_new_session'] is True
    assert kwargs['env']['HF_HUB_OFFLINE']=='1'
    assert 'qspec_hybrid.lineage' in args
    assert 'pilot' not in args and 'full' not in args


def test_background_preparation_reuses_ready_run_without_starting_process(parent,monkeypatch):
    child=lineage.prepare_fresh_lineage(parent)
    monkeypatch.setattr(subprocess,'Popen',lambda *args,**kw:pytest.fail('no process should start'))
    assert lineage.start_preparation(parent)=={'state':'ready','run_root':str(child)}


def test_restored_lineage_is_not_nested_or_reset(parent):
    child=lineage.prepare_fresh_lineage(parent)
    assert lineage.prepare_fresh_lineage(child)==child
    assert not (child/'runs').exists()


def test_stop_includes_the_background_preparation_group(parent,monkeypatch):
    core.atomic_json(parent/'lineage_job.json',{'pid':987654321,'birth':'known'})
    monkeypatch.setattr(core,'alive',lambda job:bool(job))
    stopped=[]
    monkeypatch.setattr(core.os,'killpg',lambda pid,sig:stopped.append(pid))
    core.stop(parent)
    assert stopped==[987654321]
