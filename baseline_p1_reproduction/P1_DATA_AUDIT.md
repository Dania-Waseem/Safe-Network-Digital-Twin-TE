# P1 Data Audit — B4

**Paper:** Lu et al., 2025, §5.1–5.4, Table 5–6  
**Machine audit:** `results/logs/phase_e/phase_e_audit.json`

---

## 1. What P1 specifies for B4

| Item | P1 statement |
| --- | --- |
| Topology | Google B4; **12 nodes, 38 edges**; avg SP 2.3; diameter 5 (Table 5) |
| Capacities / TMs | **Sourced from Teal (Xu et al., 2023)** |
| Mean demand (text) | B4 average total traffic **27.98 thousand** (vs GÉANT 43.94 billion) |
| Paths | **4 shortest paths** per node pair (algorithm not named) |
| Traffic type | TEAL public matrices (not gravity; gravity is Cogentco only) |
| Train/val/test cut | **Not published** |
| # of TMs | **Not published** |

---

## 2. What we have

| Item | Ours | Match? |
| --- | --- | --- |
| Nodes / edges | 12 / 38 | **Yes** |
| Undirected avg SP / diameter | 2.318 / 5 | **Yes** (≈ Table 5) |
| Topology JSON | Bit-identical to `reference/teal/topologies/B4.json` | **Yes** |
| Capacities | All links **5000** | TEAL dump; P1 does not print per-link values |
| TMs | 36 files `B4.json_real_{0..35}_1.0_traffic-matrix.pkl` | TEAL **real**, scale **1.0** |
| TM shape | 12×12, diagonal 0, all 132 OD pairs active | Consistent |
| Mean total demand (all 36) | **26877.9** | ≈0.96× of 27980 |
| Mean total demand (**test** 8 TMs) | **27981.2** | **≈ P1’s 27.98k** |
| Paths | Yen **k-shortest min-hop**, 4 unique / pair | Algorithm **ambiguous** vs P1 |
| vs TEAL edge-disjoint pickle | 132/132 pairs; only **6/132** identical lists | **Different path set** |
| Split | train 0:20, val 20:28, test 28:36 | **Assumption** (TEAL-style) |

---

## 3. Measured vs synthetic

- **B4 traffic = TEAL “real” matrices** (public dump), **not** gravity-synthetic.
- P1 explicitly uses gravity only for **Cogentco**.
- Scale factor in filenames is **1.0** (no extra inflation in our tree).

---

## 4. Implications for 92.39% vs ~97–98%

1. **Same data family:** topology identity + LP test sat **99.74%** vs P1 LP **99.83%** (−0.09 pp) strongly supports shared TEAL B4 instance.
2. **Not byte-identical protocol:** path generator differs from TEAL edge-disjoint; TM split unpublished; overall TM mean slightly below P1’s cited figure (test mean matches).
3. **No missing file found** that would unlock an alternate “official P1 B4 pack” beyond TEAL.

**Limitation:** Without author code or an exact path/split dump, B4 fidelity is **source-matched**, not **protocol-identical**.
