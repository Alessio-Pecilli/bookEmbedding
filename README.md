# Weighted Fixed-Order Book Embedding

The project compares QAOA (pytket + Qiskit Aer) with a CP-SAT baseline for
weighted page assignments. The encoding is fixed everywhere:

```text
qubit = edge_idx * NUM_PAGES + page
```

Counts are canonicalized once in `qaoa_solver.backend_key_to_logical_bitstring`.
For the current pytket Aer adapter, `Result.get_counts()` is empirically a tuple
`(c[0], ..., c[n-1])`; Qiskit's textual form is MSB-first and is reversed only
inside that conversion function. Energy estimation, decoding, metrics, and
tests consume only the canonical `q0, q1, ...` string.

The one-hot Hamiltonian uses

```text
alpha * sum_e (sum_p x[e,p] - 1)^2
 + beta * sum_(e,f,w) sum_p w*x[e,p]*x[f,p]
```

`ALPHA` is the configured minimum. For each instance, the model uses an
effective penalty strictly larger than `BETA * sum(w)`, which is a sufficient
bound because a valid solution has cost at most that sum while an invalid
one-hot assignment pays at least one penalty. This prevents invalid states from
being preferred by the Hamiltonian; the effective value is saved in results.

Run validation with:

```bash
pytest -q
```
