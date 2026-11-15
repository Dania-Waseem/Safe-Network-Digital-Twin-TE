# Final P1 Data Audit

| Topology | P1 source | Public artifact used | TM type | Size vs Table 5 |
| --- | --- | --- | --- | --- |
| B4 | Teal | `reference/teal` JSON + `real_*_1.0` pkl | Yates-mapped **real** (not MS WAN) | **Match** 12/38 |
| UsCarrier | Teal | TEAL JSON + **toy** pkl | All entries **1.0** | **Match** 158/378 |
| Kdl | Teal | TEAL JSON + toy | All **1.0** | **Match** 754/1790 |
| ASN | Teal | `ASN2k.json` + toy | All **1.0** | **Match** 1739/8558 |
| Cogentco | Topology Zoo + gravity | Zoo GML → 197/486 | Reconstructed gravity | **Match** |
| GÉANT | FIGRET / Uhlig | Zoo **Geant2012** 40/122 | Reconstructed gravity | **Mismatch** (P1 22/72) |

B4 test mean demand **27981** vs P1 **27980**. Capacities B4: all 5000 (TEAL).

**Cannot obtain without fabrication:** P1’s exact GÉANT 22-node TM pack; Microsoft WAN traces; unpublished gravity constants; unpublished TM split.
