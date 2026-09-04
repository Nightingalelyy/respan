# Respan Collector: the upstream OpenTelemetry Collector (contrib distribution)
# with the Respan configuration baked in. No custom daemon, no custom code —
# robustness and security updates come from upstream.
#
#   docker build -t respanai/collector:0.160.0 collector/
#   docker run -d --name respan-collector --restart unless-stopped \
#     -e RESPAN_API_KEY \
#     -p 127.0.0.1:4318:4318 -p 127.0.0.1:13133:13133 -p 127.0.0.1:8888:8888 \
#     -v respan-collector-queue:/var/lib/otelcol/respan \
#     respanai/collector:0.160.0

# The contrib image has no shell, so the writable queue directory is prepared in
# a helper stage and copied in with the collector's uid.
FROM alpine:3.20 AS prep
RUN mkdir -p /queue

FROM otel/opentelemetry-collector-contrib:0.160.0

COPY config.yaml /etc/otelcol-contrib/config.yaml
COPY --from=prep --chown=10001:10001 /queue /var/lib/otelcol/respan

# Inside a container the listeners must bind all interfaces; publish the ports
# on 127.0.0.1 only (see docker-compose.yaml) to keep them local.
ENV RESPAN_COLLECTOR_LISTEN=0.0.0.0:4318 \
    RESPAN_COLLECTOR_HEALTH_ENDPOINT=0.0.0.0:13133 \
    RESPAN_COLLECTOR_METRICS_HOST=0.0.0.0 \
    RESPAN_COLLECTOR_DATA_DIR=/var/lib/otelcol/respan

VOLUME ["/var/lib/otelcol/respan"]
EXPOSE 4318 13133 8888

# The upstream image already runs as uid 10001 with
# CMD ["--config", "/etc/otelcol-contrib/config.yaml"].
