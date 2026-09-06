# Expression coverage

## Inferred types, function bodies, and coverage file selection

```scrut {output_stream: stdout}
$ mkdir -p "$TMPDIR/expressions" && \
> printf 'search-path = ["."]\n[coverage]\nexcludes = ["ignored.py"]\n' > "$TMPDIR/expressions/pyrefly.toml" && \
> printf 'x = 1\ny = x + 2\n' > "$TMPDIR/expressions/inferred.py" && \
> printf 'from typing import Any\ndef _f(x: Any):\n    return x\n' > "$TMPDIR/expressions/dynamic.py" && \
> printf 'x = 1\n' > "$TMPDIR/expressions/ignored.py" && \
> $PYREFLY coverage expressions -c "$TMPDIR/expressions/pyrefly.toml" > "$TMPDIR/expressions.json" && \
> $JQ -e '{schema_version, summary} == {schema_version: "0.1", summary: {n_modules: 2, n_expressions: 5, n_any: 1, n_unanalyzed: 0, coverage: 80}}' "$TMPDIR/expressions.json"
true
```

## Source expressions are measured even when a stub exists

```scrut {output_stream: stdout}
$ printf 'x: int\ny: int\n' > "$TMPDIR/expressions/inferred.pyi" && \
> $PYREFLY coverage expressions -c "$TMPDIR/expressions/pyrefly.toml" > "$TMPDIR/expressions.json" && \
> $JQ -e '.summary == {n_modules: 3, n_expressions: 5, n_any: 1, n_unanalyzed: 0, coverage: 80}' "$TMPDIR/expressions.json"
true
```
