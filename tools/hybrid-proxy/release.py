"""Bounded transport retries and credential-free diagnostics for session release."""
import http.client
import re
import time


def release_retry(action):
    for attempt in range(3):
        try:
            return action()
        except Exception as error:
            status = getattr(error, 'code', getattr(error, 'status', None))
            if status == 404:
                if hasattr(error, 'close'):
                    error.close()
                return None
            if status is not None or not isinstance(error, (OSError, http.client.HTTPException)) or attempt == 2:
                raise
            time.sleep(.5 * 2**attempt)


def release_diagnostic(error, secrets):
    message = str(error)
    for secret in sorted(set(value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, '[redacted]')
    message = re.sub(r'https?://[^\s<>]+', '[endpoint]', message)
    message = re.sub(r'(?i)bearer\s+\S+', 'Bearer [redacted]', message)
    return dict(exception_type=type(error).__name__, message=message[:1024])
