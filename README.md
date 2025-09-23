# 🔐 Secure FL

This repository contains the implementation, experiments, and documentation for our research project:

**“Dual-Verifiable Framework for Federated Learning using Zero-Knowledge Proofs (ZKPs)”**

We explore how **zk-STARKs** (client-side) and **zk-SNARKs** (server-side) can be integrated into federated learning systems to provide **dual verifiability** of training and aggregation, with on-chain verification for public auditability.

---

## 📌 Contributors
- [@krishantt](https://github.com/krishantt) 
- [@bigya01](https://github.com/bigya01) 

---

## 📂 Repository Structure (planned)
```
secure-fl/
├── docs/           # Documentation, design notes, reports
├── fl/             # Federated learning baseline (Flower + PyTorch)
├── proofs/         # ZKP circuits and integrations (Cairo, Circom)
├── blockchain/     # Smart contracts and blockchain verification
├── experiments/    # Jupyter notebooks, datasets, benchmarks
├── k8s/            # Kubernetes deployment manifests
├── infra/          # OpenTofu/Terraform IaC configs
├── .gitignore
└── README.md
```
