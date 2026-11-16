# SNDlib source files: Abilene and Germany50

The files in `abilene/` and `germany50/` are stored exactly as downloaded from SNDlib (Survivable Network Design
Library, Poznan University of Technology). They were downloaded on 2026-11-16 from
`https://sndlib.put.poznan.pl/download/` (page `https://sndlib.put.poznan.pl/dynamicmatrices.overview.action`,
networks `abilene` and `germany50`). The native (plain-text) format is used; the XML network files are kept for
reference.

| File | Source URL | Bytes | SHA256 |
|---|---|---:|---|
| `abilene/abilene.txt` | https://sndlib.put.poznan.pl/download/sndlib-networks-native/abilene.txt | 9,495 | `cf951d6a6c007eb54f85518dc7b846ab15507d8b3216a26f4a4af4d465054db1` |
| `abilene/abilene.xml` | https://sndlib.put.poznan.pl/download/sndlib-networks-xml/abilene.xml | 24,634 | `a5793e17bb72ff13fed5e7d80e9a46c6af2041c4cc2e6381906bd06da8936fe3` |
| `abilene/abilene-rm.txt` | https://sndlib.put.poznan.pl/download/abilene-rm.txt | 6,161 | `2864b26d747126353a3378f771797695d22035fbdb296c03f48b2e6ee95ffd3a` |
| `abilene/directed-abilene-zhang-5min-over-6months-ALL-native.tgz` | https://sndlib.put.poznan.pl/download/directed-abilene-zhang-5min-over-6months-ALL-native.tgz | 49,203,514 | `2f311130d77e40db88da1aa6db8055b6fce8d077bf4bae87398563e1b84e70ce` |
| `germany50/germany50.txt` | https://sndlib.put.poznan.pl/download/sndlib-networks-native/germany50.txt | 46,002 | `a295adf283f42c5cce8cd5f9c2fbd8b3071bd02c011c841726c0ec29c744c1fe` |
| `germany50/germany50.xml` | https://sndlib.put.poznan.pl/download/sndlib-networks-xml/germany50.xml | 116,661 | `9dcb4fe4a213e7c2f993263d4f387468a68f83d9f723247cf606984e87f347e2` |
| `germany50/germany50-rm.txt` | https://sndlib.put.poznan.pl/download/germany50-rm.txt | 3,477 | `c8b9b998c25c224e8c416762a229a57551172dc86d6b61de2aa9708fe54c1f8b` |
| `germany50/directed-germany50-DFN-aggregated-5min-over-1day-native.tgz` | https://sndlib.put.poznan.pl/download/directed-germany50-DFN-aggregated-5min-over-1day-native.tgz | 5,739,063 | `5676e11b5b11900efdc3b9de60ad5f43f982240c74ec141d77dd0f9ed0f01fd3` |
| `germany50/directed-germany50-DFN-aggregated-1day-over-1month-native.tgz` | https://sndlib.put.poznan.pl/download/directed-germany50-DFN-aggregated-1day-over-1month-native.tgz | 748,588 | `b9e890ea1891ed96425ee654b98bf53608d6a71dee73227bcd841d6160a2bc2b` |
| `germany50/directed-germany50-DFN-aggregated-1month-over-1year-native.tgz` | https://sndlib.put.poznan.pl/download/directed-germany50-DFN-aggregated-1month-over-1year-native.tgz | 396,082 | `b88a31e40020bfdfcbfc3a9c423434692c8e1019e844de7956d5afb0a6633bfb` |

- The network files (`abilene.txt`, `germany50.txt`) give nodes, undirected links with pre-installed capacity
  and installable modules, and one static demand matrix each. The static demand matrices are not used.
- The dynamic archives hold one native-format file per traffic matrix (a network without links plus a META
  section with time stamp, granularity and unit). The experiments use the Abilene five-minute archive
  (48,096 matrices) and the Germany50 five-minute archive (288 matrices); the two other Germany50 archives
  (daily and monthly aggregates) were downloaded for the quality checks only.
- Data checks, unit conversion, capacity choice and sampling are described in `QUALITY_REPORT.md`.

## Related file used for the overlap check

`data/raw/yates/abilene.txt` and `data/raw/yates/abilene.dot` were downloaded on 2026-11-16 from the Yates
repository (`https://raw.githubusercontent.com/cornell-netlab/yates/master/data/demands/actual/abilene.txt` and
`.../data/topologies/abilene.dot`, commit `926bd89e75d884253c41fbe99c9b5eaf7ad3065b` of branch `master`).
TEAL documents its B4 `real` traffic matrices as these Yates demands mapped onto B4.

| File | Bytes | SHA256 |
|---|---:|---|
| `data/raw/yates/abilene.txt` | 66,721 | `7614bc787b13031fe0613366b3d9d4f1b3c5bb4b9df9b58d6438529ba0519a7e` |
| `data/raw/yates/abilene.dot` | 4,868 | `0cb5229cba1e883acb35b57e83ecb1b0f2066acdfe958fe73d237c7ca7ca8321` |

## Citations

- SNDlib: S. Orlowski, M. Pióro, A. Tomaszewski and R. Wessäly, "SNDlib 1.0 — Survivable Network Design
  Library", *Networks* 55(3):276–286, 2010, doi:10.1002/net.20371.
- Dynamic traffic matrices, as requested by SNDlib for publications using them: F. Idzikowski, S. Orlowski,
  C. Raack, H. Woesner and A. Wolisz, "Dynamic routing at different layers in IP-over-WDM networks —
  Maximizing energy savings", *Optical Switching and Networking*, Special Issue on Green Communications, 2011;
  and A. M. C. A. Koster, M. Kutschka and C. Raack, "Towards Robust Network Design using Integer Linear
  Programming Techniques", *Proceedings of NGI 2010*, Paris, 2010.
- Abilene traffic: the SNDlib readme names the original source as the Abilene traffic matrices of Yin Zhang
  (University of Texas at Austin), `http://www.cs.utexas.edu/~yzhang/research/AbileneTM/`, computed from
  Abilene NetFlow accounting data from 1 March to 10 September 2004. The associated method paper is
  Y. Zhang, M. Roughan, N. Duffield and A. Greenberg, "Fast accurate computation of large-scale IP traffic
  matrices from link loads", *ACM SIGMETRICS* 2003.
- Germany50 traffic: DFN-Verein (German Research Network, `http://www.dfn.de/`) accounting data from 2004–2005,
  aggregated by SNDlib to the 50 germany50 nodes. The germany50 topology was contributed to SNDlib by
  T-Systems International (R. Hülsermann).

SNDlib states that the data may contain errors from the original measurements or from the conversion; the
checks applied here are in `QUALITY_REPORT.md`.
