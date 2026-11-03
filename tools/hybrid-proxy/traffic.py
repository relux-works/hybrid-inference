"""Pre-dispatch application traffic reservations for a metered calibration."""
from contextlib import contextmanager
import threading


class TrafficBudgetExceeded(Exception):
    pass


class TrafficBudget:
    cleanup_headroom = 32 * 1024**2 + 4 * 65536
    allowance = 65536  # HTTP framing allowance per round trip; TCP/SSH retransmission is not measurable here.

    def __init__(self, limit, geometry):
        if type(limit) is not int or limit <= 0:
            raise ValueError('invalid calibration traffic budget')
        self.limit, self.geometry = limit, geometry
        self.used = self.reserved = self.requests = self.refusals = 0
        self.lock = threading.Lock()

    def bound(self, path, data):
        bound = 8 * 1024**2
        if path.endswith('/extend') and isinstance(data, dict) and data.get('return_delta'):
            fixed = data.get('chunk_index', 0) + 1 == data.get('chunk_count', 1)
            state = self.geometry['state_bytes'] + self.geometry.get('window_bytes', 0) if fixed else 0
            bound = max(bound, (self.geometry['kv_bytes'] + 32) * len(data['tokens']) + state + 1024**2 + 52)
        return bound

    @contextmanager
    def reserve(self, path, data, outgoing_bytes, method="GET"):
        response_bound = self.bound(path, data)
        charge = outgoing_bytes + response_bound + self.allowance
        with self.lock:
            available = self.limit if method == "DELETE" or path.endswith("/trace-flush") else max(0, self.limit-self.cleanup_headroom)
            if self.used + self.reserved + charge > available:
                self.refusals += 1
                raise TrafficBudgetExceeded('calibration traffic budget refused before dispatch')
            self.reserved += charge
        receipt = dict(response_limit=response_bound, actual=None)
        try:
            yield receipt
        finally:
            # An interrupted/unknown response consumes its full reservation, so retries cannot evade the cap.
            actual = charge if receipt['actual'] is None else outgoing_bytes + receipt['actual'] + self.allowance
            with self.lock:
                self.reserved -= charge
                self.used += actual
                self.requests += 1
                if actual > charge or self.used + self.reserved > self.limit:
                    raise TrafficBudgetExceeded('calibration response exceeded reserved bound')

    def status(self):
        with self.lock:
            return dict(limit_bytes=self.limit, charged_bytes=self.used, outstanding_bytes=self.reserved,
                        requests=self.requests, refused_before_dispatch=self.refusals,
                        accounting='request/response bodies plus 64KiB HTTP allowance per exchange; failed exchanges charged at bound; excludes unbounded TCP/SSH retransmissions')
