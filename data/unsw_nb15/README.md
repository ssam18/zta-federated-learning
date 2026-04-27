# UNSW-NB15 Dataset

## Overview

This directory contains network intrusion records collected at the University of New South Wales
(UNSW) Canberra Cyber Range Laboratory. The dataset was assembled using the IXIA PerfectStorm
traffic generator to produce realistic normal activity patterns, combined with attack traffic
executed by security researchers on an isolated testbed network.

## Collection Environment

- **Facility**: UNSW Canberra Cyber Range
- **Network topology**: Three-tier architecture (Internet, DMZ, internal LAN segments)
- **Capture duration**: Two capture windows (31 Jan – 2 Feb 2015; 14–16 Feb 2015)
- **Tools**: Argus, Bro-IDS for flow extraction; tcpdump for raw pcap
- **Feature extraction**: UNSW's custom feature extractor producing 49 features per record

## Attack Categories

| Category | Description |
|----------|-------------|
| Normal | Legitimate network activity from the cyber range testbed |
| Generic | Generic attack patterns not tied to a specific exploit |
| Exploits | Known CVE-based exploitation attempts |
| Fuzzers | Malformed or randomised packet injection |
| DoS | Denial-of-service flooding and resource exhaustion |
| Reconnaissance | Network scanning, probing, and enumeration |
| Backdoor | Persistent access tools and reverse shell traffic |
| Analysis | File and network analysis using attack tools |
| Shellcode | Shellcode injection payload traffic |
| Worms | Automated worm propagation and scanning |

## Files

| File | Description |
|------|-------------|
| `raw/intrusion_records.csv` | Raw connection records with 49 network features |
| `processed/` | Normalised features and integer-encoded labels for training |

The CSV in this directory is a small public sample used for unit-testing
and quick reproduction.  The full UNSW-NB15 corpus (~100 GB pcap +
~2.5 million labelled flows) must be downloaded separately from the source
below.

## Download (Full Dataset)

The dataset is distributed by the University of New South Wales Canberra
Cyber group:

- **Landing page**: [UNSW-NB15 (UNSW Canberra)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Direct file archive**: [UNSW-NB15 file index](https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys)

After download, place the combined feature CSV at
`raw/intrusion_records.csv` (or update the path passed to
`load_unsw_nb15(...)`).  The loader in `src/utils/data_loader.py` documents
the expected column layout.

## Feature Set

The 49 features span network-layer statistics and connection-level attributes:
- Flow identifiers (source/destination IP, port, protocol, state)
- Volume metrics (bytes, packets, load, window size, jitter)
- Connection timing (duration, inter-packet gaps, RTT, SYN/ACK timing)
- Service and state classification flags
- Time-window aggregation counters (e.g., `ct_srv_src`, `ct_dst_ltm`)

## Citation

> N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection
> Systems," in *Proc. MilCIS*, 2015, pp. 1–6.
