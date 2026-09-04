import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  COLLECTOR_CONFIG_TEMPLATE,
  COLLECTOR_IMAGE,
  COLLECTOR_VERSION,
  CONTAINER_NAME,
  DEFAULT_COLLECTOR_PORT,
  DEFAULT_HEALTH_PORT,
  DEFAULT_MAX_SIZE_BYTES,
  DEFAULT_METRICS_PORT,
  DEFAULT_QUEUE_SIZE,
  VOLUME_NAME,
  binaryDownloadUrl,
  buildCollectorConfig,
  checksumUrl,
  collectorBaseUrl,
  collectorPaths,
  dockerRunArgs,
  dockerVolumeInitArgs,
  exportUrlFromBaseUrl,
  formatBytes,
  parsePrometheusText,
  parseSha256,
  summarizeCollectorMetrics,
} from '../dist/lib/collector.js';

const here = path.dirname(fileURLToPath(import.meta.url));
// javascript-sdks/respan-cli/tests -> <repo>/collector/config.yaml
const REPO_CONFIG_PATH = path.resolve(here, '..', '..', '..', 'collector', 'config.yaml');

test('embedded collector config is byte-identical to collector/config.yaml', () => {
  const repoConfig = fs.readFileSync(REPO_CONFIG_PATH, 'utf-8');
  assert.equal(COLLECTOR_CONFIG_TEMPLATE, repoConfig);
  // The template is the source of the env placeholders the commands rely on.
  for (const name of [
    'RESPAN_COLLECTOR_DATA_DIR',
    'RESPAN_COLLECTOR_MAX_SIZE_BYTES',
    'RESPAN_COLLECTOR_HEALTH_ENDPOINT',
    'RESPAN_COLLECTOR_LISTEN',
    'RESPAN_COLLECTOR_MEMORY_LIMIT_MIB',
    'RESPAN_COLLECTOR_EXPORT_URL',
    'RESPAN_COLLECTOR_QUEUE_SIZE',
    'RESPAN_COLLECTOR_METRICS_HOST',
  ]) {
    assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`\${env:${name}:-`), `template has ${name}`);
  }
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes('Authorization: "Bearer ${env:RESPAN_API_KEY}"'));
});

test('constants match the collector packaging', () => {
  assert.equal(COLLECTOR_VERSION, '0.160.0');
  assert.equal(COLLECTOR_IMAGE, 'otel/opentelemetry-collector-contrib:0.160.0');
  assert.equal(DEFAULT_COLLECTOR_PORT, 4318);
  assert.equal(DEFAULT_HEALTH_PORT, 13133);
  assert.equal(DEFAULT_METRICS_PORT, 8888);
  assert.equal(DEFAULT_MAX_SIZE_BYTES, 536870912);
  assert.equal(DEFAULT_QUEUE_SIZE, 10000);
  assert.equal(CONTAINER_NAME, 'respan-collector');
  assert.equal(VOLUME_NAME, 'respan-collector-queue');
  // The template defaults agree with the constants.
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`:-${DEFAULT_MAX_SIZE_BYTES}}`));
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`:-${DEFAULT_QUEUE_SIZE}}`));
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`:-127.0.0.1:${DEFAULT_COLLECTOR_PORT}}`));
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`:-127.0.0.1:${DEFAULT_HEALTH_PORT}}`));
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`port: ${DEFAULT_METRICS_PORT}`));
  assert.ok(COLLECTOR_CONFIG_TEMPLATE.includes(`Validated against otelcol-contrib ${COLLECTOR_VERSION}`));
});

test('buildCollectorConfig with no overrides is the template', () => {
  assert.equal(buildCollectorConfig(), COLLECTOR_CONFIG_TEMPLATE);
  assert.equal(buildCollectorConfig({}), COLLECTOR_CONFIG_TEMPLATE);
  assert.equal(buildCollectorConfig({ exportUrl: undefined, dataDir: '' }), COLLECTOR_CONFIG_TEMPLATE);
});

test('buildCollectorConfig substitutes only the given placeholders and never the API key', () => {
  const yaml = buildCollectorConfig({
    exportUrl: 'https://respan.example.com/api/v2/traces',
    dataDir: '/home/me/.respan/collector/data',
    maxSizeBytes: 1073741824,
    queueSize: 20000,
    listen: '127.0.0.1:4319',
  });

  assert.match(yaml, /^    traces_endpoint: https:\/\/respan\.example\.com\/api\/v2\/traces$/m);
  // The data dir placeholder appears twice (storage + compaction directory).
  assert.equal(yaml.match(/directory: \/home\/me\/\.respan\/collector\/data$/gm).length, 2);
  assert.match(yaml, /^    max_size: 1073741824$/m);
  assert.match(yaml, /^      queue_size: 20000$/m);
  assert.match(yaml, /^        endpoint: 127\.0\.0\.1:4319$/m);
  for (const replaced of [
    'RESPAN_COLLECTOR_EXPORT_URL',
    'RESPAN_COLLECTOR_DATA_DIR',
    'RESPAN_COLLECTOR_MAX_SIZE_BYTES',
    'RESPAN_COLLECTOR_QUEUE_SIZE',
    'RESPAN_COLLECTOR_LISTEN',
  ]) {
    assert.equal(yaml.includes(`\${env:${replaced}`), false, `${replaced} replaced`);
  }
  // Untouched options keep their env placeholder (still overridable at runtime).
  assert.ok(yaml.includes('${env:RESPAN_COLLECTOR_HEALTH_ENDPOINT:-127.0.0.1:13133}'));
  assert.ok(yaml.includes('${env:RESPAN_COLLECTOR_METRICS_HOST:-127.0.0.1}'));
  assert.ok(yaml.includes('${env:RESPAN_COLLECTOR_MEMORY_LIMIT_MIB:-512}'));
  // The secret is only ever an env reference.
  assert.ok(yaml.includes('Authorization: "Bearer ${env:RESPAN_API_KEY}"'));
  assert.equal(yaml.split('${env:RESPAN_API_KEY}').length, 2);

  const rest = buildCollectorConfig({ healthEndpoint: '127.0.0.1:13134', metricsHost: '0.0.0.0', memoryLimitMib: 256 });
  assert.match(rest, /^    endpoint: 127\.0\.0\.1:13134$/m);
  assert.match(rest, /^                host: 0\.0\.0\.0$/m);
  assert.match(rest, /^    limit_mib: 256$/m);
  assert.ok(rest.includes('${env:RESPAN_COLLECTOR_EXPORT_URL:-https://api.respan.ai/api/v2/traces}'));
});

test('collectorPaths live under ~/.respan/collector', () => {
  const paths = collectorPaths('/home/me');
  const root = path.join('/home/me', '.respan', 'collector');
  assert.deepEqual(paths, {
    root,
    configPath: path.join(root, 'config.yaml'),
    dataDir: path.join(root, 'data'),
    binDir: path.join(root, 'bin'),
    binaryPath: path.join(root, 'bin', 'otelcol-contrib'),
    pidPath: path.join(root, 'collector.pid'),
    logPath: path.join(root, 'collector.log'),
  });
  assert.ok(collectorPaths().root.endsWith(path.join('.respan', 'collector')));
});

test('exportUrlFromBaseUrl mirrors the SDK base URL handling', () => {
  const expected = 'https://api.respan.ai/api/v2/traces';
  assert.equal(exportUrlFromBaseUrl('https://api.respan.ai'), expected);
  assert.equal(exportUrlFromBaseUrl('https://api.respan.ai/'), expected);
  assert.equal(exportUrlFromBaseUrl('https://api.respan.ai/api'), expected);
  assert.equal(exportUrlFromBaseUrl('https://api.respan.ai/api/'), expected);
  assert.equal(exportUrlFromBaseUrl('https://api.respan.ai/api///'), expected);
  assert.equal(exportUrlFromBaseUrl(' https://api.respan.ai/api '), expected);
  assert.equal(exportUrlFromBaseUrl('https://endpoint.respan.ai/api'), 'https://endpoint.respan.ai/api/v2/traces');
  assert.equal(exportUrlFromBaseUrl('http://localhost:8000'), 'http://localhost:8000/api/v2/traces');
  assert.equal(collectorBaseUrl(), 'http://127.0.0.1:4318');
  assert.equal(collectorBaseUrl(4319), 'http://127.0.0.1:4319');
});

test('dockerRunArgs binds to loopback, passes the key through the env, and mounts config + queue', () => {
  const opts = {
    configPath: '/home/me/.respan/collector/config.yaml',
    port: 4319,
    exportUrl: 'https://api.respan.ai/api/v2/traces',
    maxSizeBytes: 536870912,
    queueSize: 10000,
  };
  const args = dockerRunArgs(opts);
  const valuesAfter = (flag) => args.filter((_, i) => i > 0 && args[i - 1] === flag);

  assert.deepEqual(args.slice(0, 6), ['run', '-d', '--name', 'respan-collector', '--restart', 'unless-stopped']);
  assert.deepEqual(valuesAfter('-p'), ['127.0.0.1:4319:4318', '127.0.0.1:13133:13133', '127.0.0.1:8888:8888']);

  const envs = valuesAfter('-e');
  assert.ok(envs.includes('RESPAN_API_KEY'), 'API key is passed through from the environment');
  assert.equal(envs.some((e) => e.startsWith('RESPAN_API_KEY=')), false, 'never as a literal value');
  assert.deepEqual(envs.filter((e) => e !== 'RESPAN_API_KEY'), [
    'RESPAN_COLLECTOR_EXPORT_URL=https://api.respan.ai/api/v2/traces',
    'RESPAN_COLLECTOR_MAX_SIZE_BYTES=536870912',
    'RESPAN_COLLECTOR_QUEUE_SIZE=10000',
    'RESPAN_COLLECTOR_LISTEN=0.0.0.0:4318',
    'RESPAN_COLLECTOR_HEALTH_ENDPOINT=0.0.0.0:13133',
    'RESPAN_COLLECTOR_METRICS_HOST=0.0.0.0',
    'RESPAN_COLLECTOR_DATA_DIR=/var/lib/otelcol/respan',
  ]);
  assert.deepEqual(valuesAfter('-v'), [
    '/home/me/.respan/collector/config.yaml:/etc/otelcol-contrib/config.yaml:ro',
    'respan-collector-queue:/var/lib/otelcol/respan',
  ]);
  assert.deepEqual(args.slice(-3), ['otel/opentelemetry-collector-contrib:0.160.0', '--config', '/etc/otelcol-contrib/config.yaml']);
  assert.equal(args.join(' ').includes('sk-'), false);

  const custom = dockerRunArgs({ ...opts, image: 'respanai/collector:0.160.0' });
  assert.equal(custom.at(-3), 'respanai/collector:0.160.0');

  assert.deepEqual(dockerVolumeInitArgs(), [
    'run', '--rm', '-v', 'respan-collector-queue:/queue', 'alpine:3.20', 'chown', '10001:10001', '/queue',
  ]);
});

test('binaryDownloadUrl maps platform/arch to the release asset names', () => {
  const base = 'https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.160.0';
  assert.equal(binaryDownloadUrl('0.160.0', 'darwin', 'arm64'), `${base}/otelcol-contrib_0.160.0_darwin_arm64.tar.gz`);
  assert.equal(binaryDownloadUrl('0.160.0', 'darwin', 'x64'), `${base}/otelcol-contrib_0.160.0_darwin_amd64.tar.gz`);
  assert.equal(binaryDownloadUrl('0.160.0', 'linux', 'x64'), `${base}/otelcol-contrib_0.160.0_linux_amd64.tar.gz`);
  assert.equal(binaryDownloadUrl('0.160.0', 'linux', 'arm64'), `${base}/otelcol-contrib_0.160.0_linux_arm64.tar.gz`);
  assert.ok(binaryDownloadUrl(undefined, 'linux', 'x64').includes(`/v${COLLECTOR_VERSION}/`));
  assert.throws(() => binaryDownloadUrl('0.160.0', 'win32', 'x64'), /win32\/x64/);
  assert.throws(() => binaryDownloadUrl('0.160.0', 'linux', 'ia32'), /linux\/ia32/);
  assert.throws(() => binaryDownloadUrl('0.160.0', 'win32', 'x64'), /--docker/);
});

test('checksumUrl is the per-asset .sha256 file and parseSha256 reads its bare digest', () => {
  assert.equal(
    checksumUrl('0.160.0', 'darwin', 'arm64'),
    `${binaryDownloadUrl('0.160.0', 'darwin', 'arm64')}.sha256`,
  );
  const digest = 'ceb5309ba16f2587dbef765d54e15c803354d038b0495b0b691e1eb9876d17c9';
  assert.equal(parseSha256(`${digest}\n`), digest);
  assert.equal(parseSha256(`${digest}  otelcol-contrib_0.160.0_darwin_arm64.tar.gz\n`), digest);
  assert.equal(parseSha256(digest.toUpperCase()), digest);
  assert.equal(parseSha256('not a checksum'), undefined);
  assert.equal(parseSha256(digest.slice(0, 63)), undefined);
});

// Scraped from otelcol-contrib 0.160.0 running collector/config.yaml with
// queue_size=1 against a sink that answered 503 and then 400. Counters carry
// no `_total` suffix in this build.
const REAL_SCRAPE = `# HELP otelcol_exporter_enqueue_failed_spans Number of spans failed to be added to the sending queue. [Alpha]
# TYPE otelcol_exporter_enqueue_failed_spans counter
otelcol_exporter_enqueue_failed_spans{exporter="otlphttp/respan"} 3
# HELP otelcol_exporter_in_flight_requests Number of in-flight requests. [Alpha]
# TYPE otelcol_exporter_in_flight_requests gauge
otelcol_exporter_in_flight_requests{data_type="traces",exporter="otlphttp/respan"} 0
# HELP otelcol_exporter_queue_capacity Fixed capacity of the retry queue (in batches). [Alpha]
# TYPE otelcol_exporter_queue_capacity gauge
otelcol_exporter_queue_capacity{data_type="traces",exporter="otlphttp/respan"} 1
# HELP otelcol_exporter_queue_size Current size of the retry queue (in batches). [Alpha]
# TYPE otelcol_exporter_queue_size gauge
otelcol_exporter_queue_size{data_type="traces",exporter="otlphttp/respan"} 1
# TYPE otelcol_exporter_send_failed_spans counter
otelcol_exporter_send_failed_spans{exporter="otlphttp/respan",server_address="127.0.0.1",server_port="9998",url_path="/api/v2/traces"} 1
# HELP otelcol_exporter_sent_spans Number of spans successfully sent to destination. [Alpha]
# TYPE otelcol_exporter_sent_spans counter
otelcol_exporter_sent_spans{exporter="otlphttp/respan",server_address="127.0.0.1",server_port="9998",url_path="/api/v2/traces"} 8
# TYPE otelcol_process_uptime counter
otelcol_process_uptime{service_instance_id="eed9e513-fcef-41f5-8395-1244ef9627ff",service_name="otelcol-contrib",service_version="0.160.0"} 42.5
# TYPE otelcol_receiver_accepted_spans counter
otelcol_receiver_accepted_spans{receiver="otlp",transport="http"} 4
# TYPE target_info gauge
target_info{service_instance_id="eed9e513-fcef-41f5-8395-1244ef9627ff",service_name="otelcol-contrib",service_version="0.160.0"} 1
`;

test('parsePrometheusText strips labels and sums series; summarize reads the 0.160.0 exporter metrics', () => {
  const metrics = parsePrometheusText(REAL_SCRAPE);
  assert.equal(metrics.get('otelcol_exporter_enqueue_failed_spans'), 3);
  assert.equal(metrics.get('otelcol_exporter_queue_capacity'), 1);
  assert.equal(metrics.get('otelcol_exporter_queue_size'), 1);
  assert.equal(metrics.get('otelcol_exporter_send_failed_spans'), 1);
  assert.equal(metrics.get('otelcol_exporter_sent_spans'), 8);
  assert.equal(metrics.get('otelcol_process_uptime'), 42.5);
  assert.equal(metrics.get('target_info'), 1);
  assert.equal(metrics.has('# TYPE'), false);

  assert.deepEqual(summarizeCollectorMetrics(metrics), {
    queueSize: 1,
    queueCapacity: 1,
    enqueueFailedSpans: 3,
    sendFailedSpans: 1,
    sentSpans: 8,
  });

  // Several series with one name are summed; NaN / non-numeric samples are skipped.
  const multi = parsePrometheusText([
    'otelcol_exporter_sent_spans{exporter="a"} 4',
    'otelcol_exporter_sent_spans{exporter="b"} 6',
    'otelcol_exporter_queue_size{exporter="a"} NaN',
    'garbage line',
    '',
  ].join('\n'));
  assert.equal(multi.get('otelcol_exporter_sent_spans'), 10);
  assert.equal(multi.has('otelcol_exporter_queue_size'), false);

  // A `_total` suffix (Prometheus counter convention) is accepted too.
  const suffixed = parsePrometheusText('otelcol_exporter_sent_spans_total{exporter="a"} 12\notelcol_exporter_send_failed_spans_total{exporter="a"} 2');
  const summary = summarizeCollectorMetrics(suffixed);
  assert.equal(summary.sentSpans, 12);
  assert.equal(summary.sendFailedSpans, 2);
  // Counters that have not been incremented are absent from the scrape and read as 0.
  assert.equal(summary.enqueueFailedSpans, 0);
  assert.deepEqual(summarizeCollectorMetrics(new Map()), {
    queueSize: 0, queueCapacity: 0, enqueueFailedSpans: 0, sendFailedSpans: 0, sentSpans: 0,
  });
});

test('formatBytes', () => {
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(1536), '1.5 KiB');
  assert.equal(formatBytes(131072), '128 KiB');
  assert.equal(formatBytes(536870912), '512 MiB');
  assert.equal(formatBytes(2.5 * 1024 ** 3), '2.5 GiB');
  assert.equal(formatBytes(-1), '-');
});
