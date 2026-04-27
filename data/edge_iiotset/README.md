# Edge-IIoTset Dataset

## Overview

This directory contains network traffic data collected from an industrial Internet of Things (IIoT)
testbed comprising Raspberry Pi 4 edge devices, programmable logic controllers (PLCs), SCADA
system components, smart sensors, and actuators. Traffic was captured across multiple industrial
communication protocols in a realistic deployment environment.

## Collection Environment

- **Edge devices**: Raspberry Pi 4 (4 GB RAM) running embedded Linux
- **Industrial hardware**: Siemens S7-1200 PLCs, custom SCADA HMI nodes, Modbus RTU/TCP slaves
- **Protocols captured**: Modbus/TCP, CoAP, MQTT (v3.1.1), HTTP/1.1, DNS, ARP, ICMP, TCP, UDP
- **Capture tool**: Wireshark 3.6 / tshark on mirrored switch ports
- **Duration**: Traffic captured over 72 hours of continuous operation

## Traffic Classes (15 total)

| Class | Description |
|-------|-------------|
| Normal | Legitimate IIoT device communication |
| DoS_TCP | TCP-based denial-of-service flooding |
| DoS_UDP | UDP flood targeting IIoT endpoints |
| Scanning | Network reconnaissance and host discovery |
| MITM_Attack | Man-in-the-middle interception on Modbus bus |
| Fingerprinting | Device fingerprinting via crafted probes |
| Password | Brute-force credential attacks on SSH/FTP |
| Port_Scanning | SYN scan and version detection sweeps |
| Ransomware | Ransomware lateral movement and C2 traffic |
| Backdoor | Backdoor installation and persistent shell activity |
| Vulnerability_Scanner | Automated vulnerability scanner probes |
| Upload | Unauthorized data exfiltration |
| SQL_Injection | SQL injection via HTTP API endpoints |
| XSS | Cross-site scripting via web HMI |
| MITM_ARP | ARP spoofing and cache poisoning |

## Files

| File | Description |
|------|-------------|
| `raw/network_traffic_samples.csv` | Packet-level feature vectors extracted from pcap captures |
| `processed/` | Normalised, label-encoded versions ready for model training |

The CSV in this directory is a small public sample used for unit-testing and
quick reproduction.  The full dataset (~14 GB of pcap and CSV) must be
downloaded separately from the source below.

## Download (Full Dataset)

The full Edge-IIoTset corpus is hosted on IEEE DataPort:

- **Landing page**: [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
- **Mirror (Kaggle, no login)**: [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
- **Companion paper (DOI 10.1109/ACCESS.2022.3165809)**: [IEEE Access](https://ieeexplore.ieee.org/document/9751703/)

After download, place the CSV at `raw/network_traffic_samples.csv` (or
update the path passed to `load_edge_iiotset(...)`).  The loader in
`src/utils/data_loader.py` documents the expected column layout.

## Feature Extraction

Features were extracted using tshark with a custom dissector profile. Each row represents one
network packet. Features include layer-2 through layer-7 fields covering Ethernet, IP, TCP, UDP,
ICMP, ARP, MQTT, CoAP, Modbus, HTTP, and DNS headers.

## Citation

If you use this data in your work, please cite the original Edge-IIoTset publication:

> M. A. Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT
> and IIoT Applications for Centralized and Federated Learning," *IEEE Access*, vol. 10,
> pp. 40281–40306, 2022.
