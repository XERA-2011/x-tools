---
description: Standard development workflow. Loads project guidelines before coding.
---

# Development Workflow

Before making any code changes, complete these preliminary steps.

## 1. Load Required Guidelines

Read the skill document:
- `.agent/skills/streamlit_standards/SKILL.md`

## 2. Confirm Key Constraints

After reading, briefly state the most critical constraints:
- "Python 3.9 compatible syntax required"
- "No fake/mock data - use empty DataFrame on error"
- "CN market: red=up, US market: green=up"

## 3. Proceed with Development

Now proceed with the coding request, ensuring all changes adhere to the guidelines.

## 4. Verification Check

Before finishing, verify:
- [ ] No Python 3.10+ syntax (`X | Y`, `dict[K, V]`)
- [ ] All functions have type hints
- [ ] Error handling uses `st.error()` not exceptions
- [ ] Cache TTL is appropriate for data type
- [ ] Missing data displays as `"--"` not `0` or `NaN`

---

**Usage**: `/dev <your coding request>`
