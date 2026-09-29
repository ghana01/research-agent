import time


class Span:
    def __init__(self, trace, name, metadata=None):
        self.trace = trace
        self.name = name
        self.metadata = metadata or {}
        self.start_time = time.perf_counter()
        self.duration = 0

    def end(self):
        self.duration = time.perf_counter() - self.start_time
        self.trace.add_span(self)


class Trace:
    def __init__(self):
        self.spans = []

    def start_span(self, name, metadata=None):
        return Span(self, name, metadata)

    def add_span(self, span):
        self.spans.append(span)

    def print_trace(self):
        print("\n========== TRACE ==========")

        for span in self.spans:
            print(f"\nSpan: {span.name}")
            print(f"Duration: {span.duration:.4f}s")

            if span.metadata:
                print("Metadata:")
                for key, value in span.metadata.items():
                    print(f"  {key}: {value}")