"""Opt-in in-memory monotonic spans. Serialization happens after the measured stream."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import itertools
import inspect
import os
import socket
import time

_domain = socket.gethostname() + ':' + str(os.getpid()) + ':python.monotonic_ns'
_current = ContextVar('hybrid_timing', default=None)
_rows = []
_ids = itertools.count(1)
_enabled = False
_limit = 1_000_000
_dropped = 0


def start(limit=1_000_000):
    global _enabled, _limit, _dropped
    _rows.clear()
    _enabled, _limit, _dropped = True, limit, 0


def stop():
    global _enabled
    _enabled = False


def rows():
    if _dropped:
        raise ValueError('timing span buffer overflow: ' + str(_dropped))
    return list(_rows)


def _emit(row):
    global _dropped
    if len(_rows) < _limit:
        _rows.append(row)
    else:
        _dropped += 1


@contextmanager
def span(name, **fields):
    if not _enabled:
        yield {}
        return
    parent = _current.get() or {}
    identity = str(next(_ids))
    row = dict(clock_domain=_domain, id=identity, parent_id=parent.get('id'), parent_domain=parent.get('clock_domain'),
               name=name, start_ns=time.monotonic_ns(), bytes=0, tokens=0,
               call_id=parent.get('call_id', identity), session_id=parent.get('session_id', ''),
               transfer_id=parent.get('transfer_id', ''), chunk_id=parent.get('chunk_id', -1),
               n0=parent.get('n0', -1), n1=parent.get('n1', -1),
               origin=parent.get('origin', 'critical'), instrumentation=name.startswith('evidence.'))
    row.update(fields)
    token = _current.set(row)
    try:
        yield row
    finally:
        row['end_ns'] = time.monotonic_ns()
        row['duration_ns'] = row['end_ns'] - row['start_ns']
        _current.reset(token)
        _emit(row)


def point(name, **fields):
    if _enabled:
        with span(name, **fields):
            pass


def enabled():
    return _enabled


def annotate(**fields):
    row = _current.get()
    if _enabled and row is not None:
        row.update(fields)


def traced(name):
    def decorate(function):
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def generate(*args, **kwargs):
                with span(name):
                    yield from function(*args, **kwargs)
            return generate
        @wraps(function)
        def run(*args, **kwargs):
            with span(name):
                return function(*args, **kwargs)
        return run
    return decorate


def context():
    return dict(_current.get() or {})


def headers():
    row = _current.get()
    if not _enabled or row is None:
        return {}
    return {'X-Hybrid-Trace-Call': row['call_id'], 'X-Hybrid-Trace-Parent': row['id'],
            'X-Hybrid-Trace-Domain': _domain, 'X-Hybrid-Trace-Origin': row['origin'],
            'X-Hybrid-Trace-Session': row['session_id'], 'X-Hybrid-Trace-Transfer': row['transfer_id'],
            **({'X-Hybrid-Trace-Chunk': str(row['chunk_id'])} if row['chunk_id'] >= 0 else {}),
            **({'X-Hybrid-Trace-N0': str(row['n0']), 'X-Hybrid-Trace-N1': str(row['n1'])} if row['n0'] >= 0 else {})}


@contextmanager
def incoming(headers):
    def safe(key):
        value = headers.get(key, '')
        return value if len(value) <= 128 and all(c.isascii() and (c.isalnum() or c in ':._-') for c in value) else ''
    parent = dict(id=safe('X-Hybrid-Trace-Parent'), clock_domain=safe('X-Hybrid-Trace-Domain'),
                  call_id=safe('X-Hybrid-Trace-Call'), origin=safe('X-Hybrid-Trace-Origin') or 'critical',
                  session_id=safe('X-Hybrid-Trace-Session'), transfer_id=safe('X-Hybrid-Trace-Transfer'))
    token = _current.set(parent)
    try:
        yield
    finally:
        _current.reset(token)


def account(spans, start_ns, first_ns, call_id, domain=_domain):
    """Subtract the union of measured leaves in one clock domain, never remote clock offsets or parents."""
    if first_ns is None:
        return dict(status='no_first_text', residual_ns=None)
    if first_ns < start_ns:
        raise ValueError('negative TTFT interval')
    selected = [r for r in spans if r['clock_domain'] == domain and r['call_id'] == call_id]
    by_id = {r['id']: r for r in selected}
    if len(by_id) != len(selected):
        raise ValueError('duplicate timing span id')
    parents = set()
    for row in selected:
        if row['end_ns'] < row['start_ns'] or row['duration_ns'] != row['end_ns'] - row['start_ns']:
            raise ValueError('invalid timing interval')
        parent = by_id.get(row['parent_id']) if row['parent_domain'] == domain else None
        if parent:
            if not parent['start_ns'] <= row['start_ns'] <= row['end_ns'] <= parent['end_ns']:
                raise ValueError('timing child outside parent')
            parents.add(parent['id'])
    leaves = [r for r in selected if r['id'] not in parents and r['name'] != 'bench.call']
    intervals = sorted((max(start_ns, r['start_ns']), min(first_ns, r['end_ns']), r['id']) for r in leaves
                       if r['start_ns'] < first_ns and r['end_ns'] > start_ns)
    total, end = 0, start_ns
    overlap = 0
    for begin, finish, _ in intervals:
        overlap += max(0, min(end, finish) - begin)
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return dict(clock_domain=domain, call_id=call_id, ttft_ns=first_ns - start_ns,
                leaf_union_ns=total, residual_ns=first_ns - start_ns - total, overlap_not_added_ns=overlap,
                leaf_ids=[r[2] for r in intervals], remote_clock_spans_not_added=True,
                rule='clipped leaf union in bench/proxy process; native remote children reported separately')
