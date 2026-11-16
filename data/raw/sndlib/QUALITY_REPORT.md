# SNDlib Abilene and Germany50: data quality report

Every number below is read from the files under `data/raw/sndlib/` by `implementation/preprocessing/sndlib_prepare.py`; the machine-readable versions are `data/processed/sndlib/<topology>_quality.json`, `<topology>_sample.json`, `<topology>_total_demand.csv`, `abilene_b4_overlap.json` and `data/processed/calibrated_load/sndlib_calibration_rows.jsonl`.

## 1. Size, time resolution and date range

| | Abilene | Germany50 |
|---|---|---|
| Nodes | 12 | 50 |
| Undirected links in the network file | 15 | 88 |
| Directed links used (each undirected link in both directions) | 30 | 176 |
| Connected (strongly, directed) | True | True |
| Dynamic archive | `directed-abilene-zhang-5min-over-6months-ALL-native.tgz` | `directed-germany50-DFN-aggregated-5min-over-1day-native.tgz` |
| Traffic matrices in the archive | 48,096 | 288 |
| Time resolution (META `granularity`) | 5min | 5min |
| First / last matrix (META `time`) | 20040301-0000 / 20040910-2355 | 20050215-0000 / 20050215-2355 |
| Calendar days with data | 167 | 1 |
| Demand unit (META `unit`) | MBITPERSEC | MBITPERSEC |

Abilene covers 1 March to 10 September 2004 with the gaps listed in the SNDlib readme (15.03–01.04, 16.04–21.04, 29.04–30.04 and 20.08.2004); 167 days × 288 five-minute bins = 48,096, which equals the number of files. Germany50's five-minute archive covers a single day (15 February 2005). It is the only Germany50 archive with at least 200 matrices; the other two archives were downloaded and checked but are not used, because they have a different aggregation period (daily and monthly averages):

| Germany50 archive | Matrices | First | Last | Granularity | Valid | Median total (Mbit/s) |
|---|---:|---|---|---|---:|---:|
| `directed-germany50-DFN-aggregated-1day-over-1month-native.tgz` | 28 | 20050201 | 20050228 | 1day | 28 | 4,824.7 |
| `directed-germany50-DFN-aggregated-1month-over-1year-native.tgz` | 14 | 200401 | 200502 | 1month | 14 | 3,863.9 |

## 2. Units and capacities

All demand values are in Mbit/s (`unit = MBITPERSEC` in the META section of every matrix). SNDlib converted the original Abilene counts (100 bytes per 5 minutes, packet sampling rate 100) with ×8/3/10⁶; the DFN byte counts were also converted to Mbit/s by SNDlib. No further conversion is applied to demands. Capacities are expressed in the same unit, Mbit/s.

| Topology | Capacity rule | Values used (Mbit/s, directed links) |
|---|---|---|
| Abilene | SNDlib pre-installed capacity of each link, applied to both directions | 2,480 × 2, 9,920 × 28 |
| Germany50 | no pre-installed capacity in the file (0.00 on all 88 links); one installable module of 40 units per link | 40,000 × 176 |

- **Abilene.** 9,920 Mbit/s is the OC-192 payload rate and 2,480 Mbit/s the OC-48 payload rate; the single OC-48 link is ATLAng–IPLSng. The files state the demand unit as Mbit/s, and the capacities are read in the same unit. The module list in the file (40,000 units per link) describes expansion options for network design and is not used.
- **Germany50.** The network file offers only installable modules (capacity 40, cost 3,290–4,150) and no installed capacity, and it states no unit for the module. The values used are therefore one module on every link, read as 40 Gbit/s = 40,000 Mbit/s. This is a modelling choice, not a measured capacity. All links get the same capacity, which is the only choice that adds no information beyond the file.
- **Why the absolute Germany50 value has no effect on the reported results.** Every experiment uses calibrated load: each matrix is multiplied by alpha = t / LP-MLU, and LP-MLU is proportional to 1/capacity, so a different capacity constant c rescales the routed traffic by the same c. The learned policy and the twins divide capacities and demands by the maximum capacity, so their inputs, the utilisations, the gate decisions, the fallback, shortest path, ECMP, the LP and the satisfaction metric are all unchanged. The only component that sees absolute values is the frozen P1 actor, whose features are raw capacities and demands (P1 eq. 5, unscaled).

## 3. Invalid and corrupted matrices

Rules (applied to every matrix, in this order; nothing is clipped or repaired):

1. NaN or infinite entry; 2. negative entry; 3. all-zero matrix; 4. total demand above 100× or below 1/100 of the median total; 5. demand record with an endpoint that is not a topology node, duplicate record, unparsed line or node section that differs from the network file; 6. an isolated single-interval spike: an entry above 100× the median of its node pair, whose previous and next five-minute values are both at most 10× that median, and which carries at least 1% of the matrix total.

| Check | Abilene | Germany50 |
|---|---:|---:|
| NaN / infinite entries | 0 | 0 |
| Negative entries | 0 | 0 |
| All-zero matrices | 0 | 0 |
| Demand endpoints not in the topology | 0 | 0 |
| Node sections identical to the network file | True | True |
| Self-demand records | 0 | 0 |
| Duplicate records / unparsed lines | 0 / 0 | 0 / 0 |
| Total demand, max / median | 4.61 | 1.53 |
| Total demand, min / median | 0.376 | 0.230 |
| Entries above 100× their pair median | 4,135 (in 3,832 matrices) | 15,536 (in 288 matrices) |
| … of which isolated in time | 3 | 2085 |
| … of which isolated and ≥ 1% of the total (rule 6) | 1 | 0 |
| **Matrices removed** | **1** | **0** |
| Valid matrices | 48,095 | 288 |

- Abilene removed: index 14924 (20040517-1940): isolated single-interval spike (>100x pair median, neighbours <=10x, >=1% of total).

Most entries above 100× their pair median are not errors. On Germany50 they sit on sparse pairs whose median is close to zero (1003 of 2,450 pairs have median 0; 379 pairs are never positive), and they carry well under 1% of the matrix total. On Abilene, 762 matrices contain an entry that is above 100× its pair median and carries at least 25% of the matrix total, on three pairs (CHINng->HSTNng: 260, CHINng->LOSAng: 471, LOSAng->HSTNng: 31 five-minute bins). Only 3 of the 4,135 Abilene entries above 100× their pair median are isolated (both neighbouring bins at most 10× the median); the others sit inside elevated periods. Counting only bins above 100×, they form 1,055 runs (longest 181 bins = 15 hours), and 48% of the entries lie in runs of at least one hour; the dominant-pair matrices cluster in 8–15 April 2004. Persistent elevated flows of this kind are the signature of long bulk transfers rather than of a counting error, so they are kept: removing them would drop real high-load periods, which are the situations a safety gate is meant to handle.

## 4. Node coverage and paths

| | Abilene | Germany50 |
|---|---:|---:|
| Demand records whose node is missing from the topology | 0 | 0 |
| Ordered pairs without a directed path | 0 | 0 |
| Candidate paths (Yen k = 4, hop count) | 522 | 9,800 |
| Pairs with fewer than 4 paths | 2 | 0 |
| Pairs with no path | 0 | 0 |
| Positive-demand pairs without a path | 0 | 0 |
| Path validation problems | 0 | 0 |
| Path backend | networkx.shortest_simple_paths | networkx.shortest_simple_paths |

Every node named in any matrix is a node of the network file (the node section of every matrix is identical to the network's), and every demand has at least one candidate path.

## 5. Traffic statistics (all valid matrices)

| | Abilene | Germany50 |
|---|---:|---:|
| Total demand, min (Mbit/s) | 968.3 | 1,150.9 |
| Total demand, 5th percentile | 1,727.4 | 2,413.2 |
| Total demand, median | 2,576.5 | 4,998.6 |
| Total demand, mean | 2,841.3 | 4,907.3 |
| Total demand, 95th percentile | 4,546.8 | 7,159.6 |
| Total demand, max | 11,889.0 | 7,649.8 |
| Daily mean total, lowest / highest day | 1,601.9 / 5,844.9 | 4,907.3 / 4,907.3 |
| Node pairs | 132 | 2450 |
| Share of traffic on the largest pair | 5.4% | 3.8% |
| Share on the largest 10% of pairs | 41.9% | 94.2% |
| Pairs carrying 80% of traffic | 48 | 86 |
| Gini coefficient of mean pair demand | 0.61 | 0.95 |
| Zero entries (share of all OD entries) | 5.2% | 41.2% |

Largest pairs by mean demand:

- Abilene: WASHng->NYCMng 154.3 Mbit/s (5.4%); LOSAng->CHINng 153.8 Mbit/s (5.4%); CHINng->LOSAng 134.7 Mbit/s (4.7%); WASHng->ATLAng 106.2 Mbit/s (3.7%); NYCMng->WASHng 94.8 Mbit/s (3.3%)
- Germany50: Stuttgart->Frankfurt 185.1 Mbit/s (3.8%); Kassel->Frankfurt 163.2 Mbit/s (3.3%); Berlin->Frankfurt 159.9 Mbit/s (3.3%); Aachen->Frankfurt 157.5 Mbit/s (3.2%); Chemnitz->Frankfurt 144.3 Mbit/s (2.9%)

Mean total demand by hour of day (Mbit/s, all valid matrices; hour as given in the META time stamp):

| Hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Abilene | 2,775 | 2,777 | 2,740 | 2,731 | 2,666 | 2,585 | 2,652 | 2,670 | 2,621 | 2,575 | 2,433 | 2,256 | 2,307 | 2,599 | 2,845 | 3,062 | 3,189 | 3,297 | 3,360 | 3,366 | 3,357 | 3,291 | 3,147 | 2,888 |
| Germany50 | 3,638 | 3,075 | 2,971 | 2,789 | 2,530 | 2,364 | 2,756 | 3,754 | 4,844 | 5,829 | 6,441 | 6,448 | 6,805 | 6,814 | 7,254 | 7,145 | 6,620 | 6,323 | 5,407 | 5,254 | 4,903 | 4,687 | 4,729 | 4,395 |

The complete total-demand time series is in `data/processed/sndlib/<topology>_total_demand.csv`.

## 6. Overlap between Abilene and the B4 traffic

The B4 traffic in this repository is TEAL's `real` set, which TEAL documents as the Yates `abilene.txt` demands mapped onto B4. The Yates file (`data/raw/yates/abilene.txt`) was compared with the SNDlib series:

- B4 matrix *i* equals Yates matrix *i* divided by 10⁵ with the diagonal set to 0 (`True` for all 36), with the node order unchanged (the only isomorphism between the Yates Abilene graph and the SNDlib Abilene graph is the identity).
- Every Yates matrix is exactly rank one (largest relative deviation from the gravity product of its own row and column sums: 1.1e-08). Yates therefore uses the *simple gravity* estimate p(s,d) = p(s)·p(d) that Zhang's archive ships alongside the measured OD values, not the measured OD matrix that SNDlib uses.
- No SNDlib matrix equals a B4 matrix, even allowing any node permutation and any scale: the sorted, sum-normalised OD vectors differ by L1 = 0.038–0.055 at best.
- The **time range overlaps**. Matching the ingress and egress shares of each Yates matrix against all 48,096 SNDlib matrices, the SNDlib slot 2004-03-01 00:00 + 5·*i* min ranks 0–38 (median 7) for Yates matrix *i*; every best match falls between 00:00 and 05:10 on 1 March 2004; the total demand of the 36 Yates matrices tracks the SNDlib totals of those 36 slots (correlation 0.96; Yates is 1.19× larger because the gravity estimate includes the diagonal).

B4's traffic is thus a gravity estimate of the Abilene network during 2004-03-01 00:00–02:55 (B4 train indices 0–19 = 00:00–01:35, validation 20–27 = 01:40–02:15, held aside 28–35 = 02:20–02:55). These 36 slots are excluded before sampling, so no Abilene test matrix comes from an interval the model was trained or validated on. The Abilene topology itself is new to the model (B4 is a different graph with 38 directed links), but its traffic comes from the same measurement campaign as the B4 training traffic. Abilene is therefore an **unseen topology with related traffic**. Germany50 (DFN, 2005) has no relation to any training data.

## 7. Sampling

The eligible matrices (valid, and for Abilene outside the B4 slots) are put in time order and split into 200 equal consecutive strata; one matrix is drawn from each stratum with numpy `default_rng(2026)`. The link-failure experiment uses 50 of these, drawn without replacement with `default_rng(2027)`. Indices and timestamps are in `data/processed/sndlib/<topology>_sample.json`.

| | Abilene | Germany50 |
|---|---:|---:|
| Eligible matrices | 48,059 | 288 |
| Sampled | 200 | 200 |
| First / last sampled | 20040301-2005 / 20040910-1555 | 20050215-0005 / 20050215-2355 |
| Distinct days in the sample | 155 | 1 |
| Hours of day covered | 24 / 24 (min 4, max 12 per hour) | 24 / 24 (min 6, max 12 per hour) |
| Median total demand, sample / all valid (Mbit/s) | 2,566.2 / 2,576.5 | 5,142.1 / 4,998.6 |
| Link-failure subset | 50 | 50 |

Germany50 has 288 valid matrices, so 200 can be drawn, but all of them come from one day: the sample spreads over the times of that day and not over different days.

## 8. Calibrated load

Each sampled matrix is rescaled to the reference MLU targets 0.5, 0.7, 0.9 and 1.1, with the k = 4 min-MLU path LP (SciPy HiGHS) as the reference, exactly as for GEANT. Native-scale MLUs of the sampled matrices:

| Topology | TMs | Shortest MLU | ECMP MLU | LP MLU | LP s/TM | alpha(0.9), median |
|---|---:|---:|---:|---:|---:|---:|
| Abilene | 200 | 0.183 | 0.207 | 0.054 | 0.01 | 21.86 |
| Germany50 | 200 | 0.027 | 0.022 | 0.015 | 0.04 | 58.43 |

At native scale both networks are lightly loaded (Abilene, with its real capacities; Germany50's native MLU depends on the chosen capacity and is not meaningful). This is why only the calibrated load levels are used for these topologies.

## 9. Preprocessed files

- `data/processed/topologies/<topology>.json`, `graph_data/<topology>_{graph.npz,bipartite.pt,validation.json}`
- `data/processed/candidate_paths/<topology>_yen4.{npz,json}`, `normalization/<topology>.{json,npz}` (denominator = maximum link capacity)
- `data/processed/splits/test/<topology>.json`, `<topology>_demand.npz` (the 200 sampled matrices, Mbit/s)
- `data/processed/calibrated_load/sndlib_scaled_tm_index.json` (alpha per matrix and target)
- `data/scenarios/link_failures/<topology>.json` (seeds 100–1000, 0–3 failed directed links, P1 §5.5 sampler)
