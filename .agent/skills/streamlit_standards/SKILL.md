---
name: Streamlit Development Standards
description: "⚠️ MANDATORY: Read before modifying code. Contains data integrity, color conventions, and Python 3.9 compatibility rules."
---

# Streamlit Development Standards

> **Priority**: This document defines project-level constraints for x-streamlit.

---

## 1. Data Integrity Policy

> ⚠️ **ABSOLUTE RULE**: No misleading data. No fake data. Ever.

### Truth Hierarchy
1. ✅ **Real Data** - Verified from AkShare
2. ✅ **Error State** - Honest "data unavailable" message
3. ❌ **Misleading Data** - Partially true but misrepresents reality
4. ❌ **Fake Data** - Fabricated numbers with no basis

### Required Patterns

```python
# ❌ FORBIDDEN: Silent fallback to fake data
def get_price():
    try:
        return fetch_real_price()
    except:
        return 100.0  # Fake fallback!

# ✅ REQUIRED: Explicit error handling with st.error()
def get_price() -> pd.DataFrame:
    try:
        return ak.some_api()
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return pd.DataFrame()  # Empty, not fake
```

### Frontend Display
```python
# ❌ FORBIDDEN: Showing 0 or fake value
value = row.get("price", 0)

# ✅ REQUIRED: Use "--" for missing data
value = row.get("price")
display = f"{value:.2f}" if pd.notna(value) else "--"
```

---

## 2. Color Semantics

| Market | Up/Positive | Down/Negative |
|:-------|:------------|:--------------|
| CN/HK  | Red 🔴      | Green 🟢      |
| US     | Green 🟢    | Red 🔴        |
| Crypto/Metals | Red 🔴 | Green 🟢    |

---

## 3. Python 3.9 Compatibility

> ⚠️ **CRITICAL**: Docker uses **Python 3.9**. Python 3.10+ syntax is FORBIDDEN!

### Forbidden Syntax
```python
# ❌ Python 3.10+ (will crash Docker)
def func(x: str | None) -> dict[str, Any]: ...

# ✅ Python 3.9 compatible
from typing import Optional, Dict, List, Any
def func(x: Optional[str]) -> Dict[str, Any]: ...
```

### Replacement Rules
| Python 3.10+ | Python 3.9 | Import |
|-------------|------------|--------|
| `X \| Y` | `Union[X, Y]` | `from typing import Union` |
| `X \| None` | `Optional[X]` | `from typing import Optional` |
| `dict[K, V]` | `Dict[K, V]` | `from typing import Dict` |
| `list[T]` | `List[T]` | `from typing import List` |

---

## 4. Type Hints

**Mandatory**: All function signatures MUST have type hints.

```python
# ✅ Good
def calculate_yield(price: float, dividend: float) -> Optional[float]: ...

# ❌ Bad
def calculate_yield(price, dividend): ...
```

---

## 5. Caching Strategy

Use Streamlit's `@st.cache_data` with appropriate TTL:

| Data Type | TTL | Example |
|-----------|-----|---------|
| Real-time (indices) | 300s (5min) | `@st.cache_data(ttl=300)` |
| Semi-static (sectors) | 600s (10min) | `@st.cache_data(ttl=600)` |
| Static (macro) | 3600s (1hr) | `@st.cache_data(ttl=3600)` |

---

## 6. Error Handling

- **Never Crash**: Functions must return empty DataFrame, not raise exceptions
- **Show Feedback**: Use `st.error()` or `st.info()` for user feedback
- **Graceful Degradation**: One failed API should NOT block the entire page

---

## ⚙️ Language Policy

> **All content in `.agent/` directory MUST be written in English.**
