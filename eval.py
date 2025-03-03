import requests
import opentelemetry.trace as trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# 1. Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# 2. Export traces to Arize Phoenix locally
exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")  # Default OpenTelemetry endpoint
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

# 3. Instrument requests globally
RequestsInstrumentor().instrument()

# 4. Set the local llama-server endpoint
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# 5. Function to send traced requests
def send_traced_request():
    with tracer.start_as_current_span("llama-server-request") as span:
        payload = {
            "model": "your-model-name",
            "messages": [{"role": "user", "content": "Hello, how are you?"}]
        }
        response = requests.post(LLAMA_SERVER_URL, json=payload)

        # Add metadata to the trace
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("model.response", response.json())

        return response.json()

# 6. Run the request
print(send_traced_request())
