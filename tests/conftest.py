import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="session")
def client():
    app.state.limiter.enabled = False
    return TestClient(app, raise_server_exceptions=False)


def to_csv(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def analyze(client):
    def _analyze(df: pd.DataFrame, **params):
        params.setdefault("data_type", "labeled")
        return client.post("/analyze", params=params, files={"file": ("data.csv", to_csv(df))})
    return _analyze


def signal_df(n=300, seed=0):
    """Labeled dataset with real signal in a few numeric features."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1.round(2), "x2": x2.round(2), "x3": x3.round(2), "y": y})
