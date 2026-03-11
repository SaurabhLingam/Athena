PROFILES = {
    "standard": {
        "missing_threshold": 0.30,
        "high_cardinality_threshold": 50,
        "correlation_threshold": 0.98,
        "anomaly_rate_threshold": 0.15,
        "skew_threshold": 1.0,
    },
    "finance": {
        "missing_threshold": 0.05,
        "high_cardinality_threshold": 100,
        "correlation_threshold": 0.95,
        "anomaly_rate_threshold": 0.05,
        "skew_threshold": 0.5,
    },
    "healthcare": {
        "missing_threshold": 0.20,
        "high_cardinality_threshold": 50,
        "correlation_threshold": 0.98,
        "anomaly_rate_threshold": 0.10,
        "skew_threshold": 1.0,
    },
    "nlp": {
        "missing_threshold": 0.50,
        "high_cardinality_threshold": 200,
        "correlation_threshold": 0.99,
        "anomaly_rate_threshold": 0.20,
        "skew_threshold": 2.0,
    },
    "custom": {
        "missing_threshold": 0.30,
        "high_cardinality_threshold": 50,
        "correlation_threshold": 0.98,
        "anomaly_rate_threshold": 0.15,
        "skew_threshold": 1.0,
    },
}


def get_profile(name: str) -> dict:
    return PROFILES.get(name.lower(), PROFILES["standard"])