# Final P1 Deviations & Unresolved Limitations

1. **ADMM numerics** — P1 unpublished argmins; P1-order solver unstable at B4’s 2 iterations; deploy uses Teal order.
2. **Path algorithm** — P1 says 4 shortest; we use Yen min-hop (not TEAL edge-disjoint; 6/132 lists match).
3. **TM split** — unpublished; 20/8/8 assumed.
4. **Grad clip / action clamps** — not in P1.
5. **UsCarrier / Kdl / ASN TMs** — public Teal is **toy unit demand**, not the proprietary mapping P1 likely used.
6. **GÉANT** — Table 5 is 22/72; obtained Zoo Geant2012 is 40/122; FIGRET pack missing.
7. **Cogentco gravity** — constants unpublished; degree-based reconstruction only.
8. **Hardware** — P1 RTX 4090; this work CPU/local.
9. **Baselines NCFlow / Figret / Teal models** — not re-run (wrappers not completed).
10. **Kdl / ASN GRL-TE** — not trained (millions of path nodes).

None of these were “fixed” by retuning toward 92.39% or 89.36%.
