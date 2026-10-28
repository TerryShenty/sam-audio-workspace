"""Bounded diagnostic ranges for a FUTURE isolated worker, never auto-installed.

Install after geometry/projection dispatch and before memo wrapping. Rebind any
dynamic aliases after the final memo wrapper. Remove outer wrappers before close.
Ranges include delegated allocation/kernel work, not exclusive device time.
Uninstall before matched throughput timing. This component performs no timing,
CUDA synchronization, checkpoint mutation, or GPU dispatch of its own.
"""
from contextlib import contextmanager
from projection_backend_candidate_v25 import packed_roles


class ProjectionProbe:
    def __init__(self, backends, blocks, *, memo=None, range_factory=None):
        if not backends or not set(backends) <= {'static', 'dynamic'}:
            raise ValueError('Explicit static/dynamic backend names required')
        if len({id(m) for m in backends.values()}) != len(backends):
            raise ValueError('Distinct backend modules required')
        self.roles = packed_roles(blocks)
        self.memo = memo
        self.active = self.closed = False
        self.report = None
        self.backends = dict(backends)
        self.originals = {name: mod.grouped_mm for name, mod in backends.items()}
        if not all(callable(fn) for fn in self.originals.values()):
            raise ValueError('Backend functions must be callable')
        if range_factory is None:
            from torch.profiler import record_function
            range_factory = record_function
        self.range_factory = range_factory
        self.wrappers = {name: self._wrap(name, fn) for name, fn in self.originals.items()}
        self._idle()
        for name, mod in self.backends.items():
            mod.grouped_mm = self.wrappers[name]

    def _idle(self):
        if self.memo is not None and self.memo.live_bytes:
            raise RuntimeError('Diagnostic boundaries require no retained memo graph')

    def _wrap(self, backend, original):
        def grouped_mm(a, p, ends, rows, transposed):
            if self.closed:
                raise RuntimeError('Closed diagnostic wrapper cannot be reused')
            if not self.active:
                return original(a, p, ends, rows, transposed)
            entry = self.roles.get(id(p))
            if entry is None or entry[0] is not p:
                raise ValueError('Unknown packed projection identity in diagnostic capture')
            if type(transposed) is not bool:
                raise ValueError('Diagnostic direction must be an explicit boolean')
            report = self.report
            if report['attempted_calls'] >= report['max_calls']:
                raise RuntimeError('Diagnostic call budget exhausted')
            direction = 'forward' if transposed else 'backward'
            label = f'qwen_nf4/{backend}/{entry[1]}/{direction}'
            counts = report['ranges'].setdefault(label, dict(attempted=0, completed=0))
            report['attempted_calls'] += 1
            counts['attempted'] += 1
            with self.range_factory(label):
                result = original(a, p, ends, rows, transposed)
            counts['completed'] += 1
            report['completed_calls'] += 1
            return result
        return grouped_mm

    @contextmanager
    def capture(self, *, max_calls=4096):
        if self.closed or self.active:
            raise RuntimeError('Capture must be non-nested on an open probe')
        if type(max_calls) is not int or not 0 < max_calls <= 10000:
            raise ValueError('Bounded diagnostic call count required')
        self._idle()
        report = dict(status='CAPTURING', max_calls=max_calls, attempted_calls=0,
                      completed_calls=0, ranges={}, diagnostic_not_throughput=True,
                      exclusive_device_time=False, timer_used=False)
        self.report = report
        self.active = True
        try:
            yield report
            self._idle()
        except BaseException:
            report['status'] = 'FAILED_DIAGNOSTIC_CAPTURE'
            raise
        else:
            report['status'] = 'COMPLETE_DIAGNOSTIC_CAPTURE'
        finally:
            self.active = False

    def close(self):
        if self.closed or self.active:
            raise RuntimeError('Cannot close an active or closed probe')
        self._idle()
        if any(mod.grouped_mm is not self.wrappers[name]
               for name, mod in self.backends.items()):
            raise RuntimeError('Remove all outer wrappers before closing diagnostic probe')
        for name, mod in self.backends.items():
            mod.grouped_mm = self.originals[name]
        self.closed = True
