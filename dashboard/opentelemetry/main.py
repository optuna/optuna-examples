import wsgiref.simple_server

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.wsgi import OpenTelemetryMiddleware
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from optuna.storages import RDBStorage
import optuna_dashboard

SQLALCHEMY_URL = "sqlite:///db.sqlite3"
OTEL_COLLECTOR_ENDPOINT = "http://127.0.0.1:4318/v1/metrics"


def main() -> None:
    resource = Resource.create({"service.name": "optuna-dashboard"})
    readers = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=OTEL_COLLECTOR_ENDPOINT),
            export_interval_millis=1000,
            export_timeout_millis=5000,
        ),
    ]
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))

    # If you want to use PrometheusMetricReader, uncomment the following lines
    # from prometheus_client import start_http_server
    # from opentelemetry.exporter.prometheus import PrometheusMetricReader
    # print("Metrics endpoint: http://127.0.0.1:9464/metrics")
    # start_http_server(port=9464, addr="127.0.0.1")
    # readers.append(PrometheusMetricReader("optuna_dashboard"))

    # If you want to see metrics in the console, uncomment the following line
    # from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
    # readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))

    # Enable opentelemetry-instrumentation-sqlalchemy
    storage = RDBStorage(SQLALCHEMY_URL, skip_compatibility_check=True, skip_table_creation=True)
    SQLAlchemyInstrumentor().instrument(
        engine=storage.engine,
        meter_provider=metrics.get_meter_provider(),
    )

    # Enable opentelemetry-instrumentation-wsgi
    app = optuna_dashboard.wsgi(storage=storage)
    app = OpenTelemetryMiddleware(app, meter_provider=metrics.get_meter_provider())

    # Start Optuna Dashboard
    with wsgiref.simple_server.make_server("127.0.0.1", 8080, app) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
