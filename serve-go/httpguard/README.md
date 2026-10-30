# HTTP API guard

This standalone package is an incremental component, not a running server or a
complete Strata security boundary. Wrap an API handler with `New(Config, handler)`.
Only canonical `/v1/` paths are forwarded; other paths are rejected. Paths must
use unescaped ASCII letters, digits, `/`, `-`, `.`, `_` or `~`, without dot
segments or repeated slashes. A single trailing slash is permitted. Escaped
characters, backslashes and semicolon path parameters are rejected, even when
they might be harmless to a particular router. Query strings are unaffected.
Install the guard before any router or middleware that rewrites the request path.
Do not use it to
protect the web UI, management routes, or a telemetry endpoint with a separate
read-only key. Those policies are not implemented.

The default per-request body limit is 256 MiB; set `MaxBody` in bytes to override it.
The guard reads a complete bounded body before invoking the handler, rejects
oversized bodies with 413 and incomplete or malformed bodies with 400, and
replaces the request body with the decoded bytes. This prevents side effects
from incomplete chunked requests. This is not an aggregate memory limit:
concurrent requests can each buffer up to `MaxBody`, with additional allocation
and handler overhead. Integrators must bound concurrent requests before this
guard (including requests being read), choose `MaxBody` and concurrency for
their memory budget, and configure finite `http.Server` `ReadHeaderTimeout` and
`ReadTimeout` values appropriate for their clients. The guard does not enforce
concurrency or timeouts itself.
The caller supplies already-normalized `HostNames`, explicit origins and
individual trimmed API keys; invalid configuration fails at construction.
Loopback names are always allowed. IP Host headers are accepted, but foreign IP
Origins are not accepted unless they match the request Host or configured names.
With no key, Host checks apply, and API POSTs with an Origin must come from an
allowed origin and use `application/json`. With keys, Host and Origin checks are
skipped, and bearer or x-api-key authentication applies to API routes.
OPTIONS is unauthenticated and emits configured CORS headers.

Known differences from Python: Go's `net/http` owns HTTP framing, header
validation, chunk-size and trailer limits, connection reuse, and error handling
before this handler. It may reject wire forms Python accepts, and does not
provide Python's exact chunk/trailer limits or early oversized chunk-size check.
It handles chunked over Content-Length according to Go's parser. The guard does
not discover bind/LAN names, read environment variables, implement Python's
unread-body drain deadline or HTTP/1.0 connection-close policy, reproduce exact
error text, expose telemetry-key access, or distinguish API routes that Python
does not authenticate. Route callers must account for these differences before
integration; this package is not a drop-in replacement for `serve/server.py`.
