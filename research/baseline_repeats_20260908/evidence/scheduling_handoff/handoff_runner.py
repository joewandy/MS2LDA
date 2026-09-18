"""One-off audited scheduling handoff; never edits frozen model source."""

import os
from pathlib import Path
import shutil
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from benchmarks.neural_ms2lda.baseline_repeats import baseline_fit_identity, require_cached_fit
from benchmarks.neural_ms2lda.utils import read_json_object, write_json, sha256_file
from scripts.run_baseline_repeats import run_stage, utc_now

ROOT = Path('/home/joewandy/Work/git/MS2LDA/output/benchmarks/baseline_repeats_20260908')
SNAPSHOT = ROOT / 'source_snapshot'
PREPARED = ROOT.parent / 'minimal_neural_20260905/prepared'
ASSETS = ROOT.parent / 'minimal_neural_20260905/assets'
AUDIT = ROOT / 'parallel_handoff'
SUPERVISOR = 434157
CHEMISTRY_LOCK = Lock()


def finished(pid):
    try:
        # Pausing the parent leaves an exited child as a zombie until resumption.
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] == 'Z'
    except FileNotFoundError:
        return True


def fit(seed, prior_gpu_pids):
    run = ROOT / 'real' / f'tomotopy_seed{seed}_attempt1'
    run_stage(AUDIT, SNAPSHOT, run, 'seal', 'scripts.prepare_msnlib_validation_view',
              ['--run', run, '--prepared-run', PREPARED])
    run_stage(AUDIT, SNAPSHOT, run, 'fit_validation', 'scripts.run_tomotopy_validation',
              ['--run', run, '--training-seed', seed])
    with CHEMISTRY_LOCK:
        while not all(finished(pid) for pid in prior_gpu_pids):
            time.sleep(5)
        run_stage(AUDIT, SNAPSHOT, run, 'chemistry', 'benchmarks.neural_ms2lda.chemical',
                  ['--run', run, '--data-root', ASSETS, '--method', 'tomotopy',
                   '--split', 'validation'])
    result = read_json_object(run / 'tomotopy/validation_only_result.json')
    cached = result['training']
    expected = baseline_fit_identity(run, seed, cached['fit_identity']['recipe'])
    require_cached_fit(cached, expected)
    for relative in ('tomotopy/model.bin', 'validation_evaluation/tomotopy/complete.json',
                     'validation_evaluation/tomotopy/validation_full_theta.npy',
                     'validation_chemical/tomotopy/complete.json'):
        if not (run / relative).is_file():
            raise RuntimeError(f'completed-cache requirement missing: {relative}')
    return {'training_seed': seed, 'run': str(run), 'completed_utc': utc_now()}


def main():
    command = Path(f'/proc/{SUPERVISOR}/cmdline').read_bytes().replace(b'\0', b' ')
    if b'scripts.run_baseline_repeats' not in command:
        raise RuntimeError('supervisor PID no longer belongs to this experiment')
    AUDIT.mkdir()
    (AUDIT / 'logs').mkdir()
    (AUDIT / 'stages').mkdir()
    shutil.copy2(__file__, AUDIT / 'handoff_runner.py')
    record = {'supervisor_pid': SUPERVISOR, 'handoff_pid': os.getpid(),
              'started_utc': utc_now(), 'status': 'pausing_supervisor',
              'model_recipe_changed': False, 'source_snapshot_changed': False,
              'runner_sha256': sha256_file(Path(__file__)),
              'reason': 'Run independent one-worker LDA fits on separate CPU cores.'}
    write_json(AUDIT / 'handoff.json', record)
    os.kill(SUPERVISOR, signal.SIGSTOP)
    running = [read_json_object(path) for path in (ROOT / 'stages').glob('*.json')]
    running = [stage for stage in running if stage['status'] == 'running']
    prior_gpu = [stage['pid'] for stage in running if 'etm_seed' in stage['run']]
    record.update(status='parallel_fits_running', paused_utc=utc_now(),
                  preserved_active_stages=running, prior_gpu_pids=prior_gpu)
    write_json(AUDIT / 'handoff.json', record)
    print('SUPERVISOR PAUSED; children preserved:', [row['pid'] for row in running], flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            tasks = [workers.submit(fit, seed, prior_gpu) for seed in (23, 42)]
            completed = [task.result() for task in tasks]
        record.update(status='parallel_fits_complete', completed=completed,
                      finished_utc=utc_now())
        write_json(AUDIT / 'handoff.json', record)
        # Both complete caches are proven before the original queue can reach
        # these seeds. Its later invocations are cache rechecks, never new fits.
        os.kill(SUPERVISOR, signal.SIGCONT)
        record.update(status='supervisor_resumed', resumed_utc=utc_now())
        write_json(AUDIT / 'handoff.json', record)
        print('SUPERVISOR RESUMED; seeds23/42 will be validated cache rechecks.', flush=True)
    except BaseException as error:
        record.update(status='handoff_failed_supervisor_still_paused', error=repr(error))
        write_json(AUDIT / 'handoff.json', record)
        raise


if __name__ == '__main__':
    main()
