# Policy sources

Place unmodified upstream visuomotor navigation code here, each in its own folder:

```
policy_sources/
  habitat_policy/        # copied from <repo_url>@<commit>
  mildenhall_policy/
```

Guidelines:
- Copy source with its original license files and headers intact.
- Record the upstream repo URL and commit/tag in a short README inside each folder.
- Avoid editing upstream files directly; instead, wrap them in `chop/models/*_wrapper.py` and register constructors via `register_policy`.
- If you must modify upstream code, note the changes in a `CHANGES.md` inside that policy folder.
