# Respan Collector

A packaged [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
that sits next to your application and forwards traces to Respan through a
**bounded, persistent queue**. It exists for one reason: long-running agents
that must not lose spans when Respan or the network is unreachable, without
an ever-growing journal on the machine.

It is the upstream `otelcol-contrib` distribution plus [`config.yaml`](config.yaml).
There is no Respan-specific daemon or code path, so robustness and security
updates come straight from OpenTelemetry.

## What it guarantees

| | Without the collector (SDK only) | With the collector |
|---|---|---|
| Blip of a few seconds | absorbed (SDK retries) | absorbed |
| Outage longer than the SDK retry window | queued spans lost | kept on disk, delivered when Respan is back |
| Application crash | in-memory queue lost | already handed to the collector, kept |
| Collector restart / host reboot | n/a | queue survives (write-ahead log) |
| Disk usage in normal operation | none | ~0 (entries are deleted on acknowledgement) |
| Disk usage during an outage | none | grows to the cap, then new batches are dropped and counted |

The queue is a buffer, not a log: an entry is removed the moment Respan
returns 2xx. Braintrust's local daemon journaled everything without a cap;
this design is the opposite.

Verified end to end against otelcol-contrib 0.160.0 with a fake Respan:
spans sent during a 503 outage were persisted, survived `kill -9` of the
collector, and were delivered after restart; a hanging backend hit the 30 s
timeout and was retried to success.

## Quick start

### Docker Compose (recommended)

```bash
export RESPAN_API_KEY=...
docker compose -f collector/docker-compose.yaml up -d
curl http://127.0.0.1:13133          # {"status":"Server available", ...}
```

### Respan CLI

```bash
respan collector start               # docker if available, otherwise the otelcol-contrib binary
respan collector status              # health, queue size, dropped/failed counters, disk usage
respan collector stop
```

### Point the SDK at it

```bash
export RESPAN_BASE_URL=http://127.0.0.1:4318
```

That is the only application-side change. The SDK posts OTLP/JSON to
`<RESPAN_BASE_URL>/api/v2/traces`; the collector receives on that exact path
and forwards to `https://api.respan.ai/api/v2/traces` with your API key.
For the pi coding agent, `respan integrate pi --with-collector` writes the
same base URL into `respan.json`.

## Configuration

Everything is an environment variable with a safe default:

| Variable | Default | Meaning |
|---|---|---|
| `RESPAN_API_KEY` | required | Sent as `Authorization: Bearer …` to Respan. Never written to the config file. |
| `RESPAN_COLLECTOR_EXPORT_URL` | `https://api.respan.ai/api/v2/traces` | Respan ingest endpoint (self-hosted / EU: your endpoint + `/api/v2/traces`). |
| `RESPAN_COLLECTOR_LISTEN` | `127.0.0.1:4318` | OTLP/HTTP receiver. The Docker image binds `0.0.0.0` inside the container; compose publishes it on `127.0.0.1` only. |
| `RESPAN_COLLECTOR_DATA_DIR` | `/var/lib/otelcol/respan` | Persistent queue directory (a named volume in compose). |
| `RESPAN_COLLECTOR_MAX_SIZE_BYTES` | `536870912` (512 MiB) | Hard cap on the queue file. Above it, new batches are dropped, never the disk filled. |
| `RESPAN_COLLECTOR_QUEUE_SIZE` | `10000` | Maximum batches kept while Respan is unreachable. |
| `RESPAN_COLLECTOR_MEMORY_LIMIT_MIB` | `512` | Collector memory limiter. |
| `RESPAN_COLLECTOR_COMPRESSION` | `none` | Export compression. Respan's ingest reads uncompressed bodies; only enable `gzip` behind a proxy that inflates it. |
| `RESPAN_COLLECTOR_HEALTH_ENDPOINT` | `127.0.0.1:13133` | Health check listener. |
| `RESPAN_COLLECTOR_METRICS_HOST` | `127.0.0.1` | Prometheus metrics host (port 8888). |

Retries never give up (`retry_on_failure.max_elapsed_time: 0`); the queue
bounds are the only limit. Exporter timeout is 30 s.

### Sizing the cap

Disk usage during an outage is `min(outage duration × ingest rate, cap)`. If
your agents produce about 5 GB of trace data a day:

| `RESPAN_COLLECTOR_MAX_SIZE_BYTES` | Outage covered before drops begin |
|---|---|
| 512 MiB (default) | ~2.5 hours |
| 2 GiB | ~10 hours |
| 5 GiB | ~1 day |

## Monitoring

Scrape `http://127.0.0.1:8888/metrics` (Prometheus format). Alert on:

| Metric | Meaning |
|---|---|
| `otelcol_exporter_queue_size` / `otelcol_exporter_queue_capacity` | batches waiting; rising means Respan is unreachable |
| `otelcol_exporter_enqueue_failed_spans` | spans dropped because the queue or disk cap was hit — data loss |
| `otelcol_exporter_send_failed_spans` | export attempts that failed (retried unless the error is permanent) |
| `otelcol_exporter_sent_spans` | spans acknowledged by Respan |

Health: `GET http://127.0.0.1:13133`.

## Duplicates on retry

Retries re-send whole batches. A batch that timed out after Respan had
already stored it therefore arrives twice. Respan's trace view shows one span
per span id; until ingest de-duplicates by span id, duplicated batches can
double-count cost and token aggregates for those runs. The 30 s exporter
timeout keeps this rare.

## Security notes

- The receiver, health and metrics listeners bind to loopback by default.
  Anything that can reach the receiver can write traces into your project.
- The API key is read from the environment by the collector process and is
  never written into `config.yaml` or the image.
- The container runs as the upstream non-root user (uid 10001).

## Files

- [`config.yaml`](config.yaml): the collector configuration (also embedded in
  `respan collector config`).
- [`Dockerfile`](Dockerfile): upstream `otel/opentelemetry-collector-contrib:0.160.0`
  plus the config and a writable queue directory.
- [`docker-compose.yaml`](docker-compose.yaml): sidecar with a named volume
  and loopback-only ports.
