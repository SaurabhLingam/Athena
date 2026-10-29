"""
Shared CSV loading for every interface (web report, MCP tools), so all of them
see the same cleaned data: placeholder strings and numeric sentinels become NaN
before column roles are inferred.

pandas is imported inside the functions so importing this module stays cheap.
"""


def read_csv_clean(source, nrows: int | None = None):
    """Read a CSV path or file object and clean placeholders.
    Returns (df, {column: {placeholder: count}})."""
    import pandas as pd
    # Read NA-like tokens as text so _replace_placeholders can count them per column
    kwargs = dict(keep_default_na=False, na_values=[""], nrows=nrows)
    try:
        df = pd.read_csv(source, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="latin1", **kwargs)
    return _replace_placeholders(df)


# "none" and "unknown" are left out: they are often real categories (e.g. "None" for no alarm)
PLACEHOLDER_TOKENS = {
    "?", "-", "--", ".", "na", "n/a", "n.a.", "n.a", "nan", "null", "#n/a", "<na>",
    "missing", "not available", "not provided", "not applicable", "not recorded", "not specified",
    "#value!", "#ref!", "#div/0!", "#num!", "#name?", "#null!", "#n/a!",
}
# A column this share numeric is numeric; its few repeated non-numeric values are placeholders
NUMERIC_SHARE_FOR_PLACEHOLDERS = 0.9
MAX_NUMERIC_PLACEHOLDER_TOKENS = 5


def _is_nines(v) -> bool:
    """999, 9999, -999, ... (three or more 9s)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == int(f) and len(str(abs(int(f)))) >= 3 and set(str(abs(int(f)))) == {"9"}


def _far_sentinels(s) -> list:
    """All-9s values that repeat and sit far outside the rest of the column."""
    candidates = [v for v, n in s.value_counts().items() if n >= 2 and _is_nines(v)]
    if not candidates:
        return []
    others = s[~s.isin(candidates)].dropna()
    if len(others) < 2:
        return []
    lo, hi = float(others.min()), float(others.max())
    span = max(hi - lo, 1.0)
    return [v for v in candidates
            if (v < 0 and (v < lo - span or lo >= 0))   # negative below, or in a non-negative column
            or (v > 0 and v > hi + 3 * span)]           # far above, so a price of 999 stays


def _replace_placeholders(df):
    """Turn placeholder strings and far-out numeric sentinels into NaN.
    Returns (df, {column: {placeholder: count}})."""
    import pandas as pd
    counts = {}
    for col in df.columns:
        s = df[col]
        found = {}
        if s.dtype == object:
            text = s.astype(str).str.strip()
            mask = s.notna() & (text.str.lower().isin(PLACEHOLDER_TOKENS) | (text == ""))
            rest = s.notna() & ~mask
            converted = pd.to_numeric(text.where(rest), errors="coerce")
            non_numeric = rest & converted.isna()
            if rest.any() and converted.notna().sum() >= NUMERIC_SHARE_FOR_PLACEHOLDERS * rest.sum():
                # Mostly numbers: a few repeated text tokens are placeholders (N.A., unk, #VALUE!)
                tokens = text[non_numeric].value_counts()
                if len(tokens) <= MAX_NUMERIC_PLACEHOLDER_TOKENS and (tokens >= 2).all():
                    mask = mask | non_numeric
            if mask.any():
                for token, n in text[mask].replace("", "<blank>").value_counts().items():
                    found[str(token)] = int(n)
                s = s.mask(mask)
                # A numeric column that only looked like text because of placeholders
                converted = pd.to_numeric(s, errors="coerce")
                if converted.notna().sum() == s.notna().sum():
                    s = converted
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            far = _far_sentinels(s)
            for v in far:
                found[str(int(v))] = int((s == v).sum())
            if far:
                s = s.mask(s.isin(far))
        if found:
            df[col] = s
            counts[str(col)] = found
    return df, counts
