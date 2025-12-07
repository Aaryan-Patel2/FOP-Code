# Docker-based GCDM-SBDD Integration for FOP Drug Discovery

This directory contains the Docker setup for running GCDM-SBDD (Geometry-Complete Diffusion for Structure-Based Drug Design) to generate molecules guided by the FOP affinity predictor.

## 🎯 Why Docker?

GCDM-SBDD has dependencies that are challenging on ARM64 (Apple Silicon) systems:
- PyTorch Geometric with CUDA support
- pytorch-scatter (requires compilation)
- Specific CUDA/cuDNN versions

Docker provides a **Linux x86_64 environment with proper GPU support**, avoiding compatibility issues while maintaining full GCDM functionality.

## 📋 Prerequisites

1. **Docker & Docker Compose** installed
   ```bash
   # Check installation
   docker --version
   docker-compose --version
   ```

2. **NVIDIA Docker Runtime** (for GPU support)
   ```bash
   # Install nvidia-docker2
   # Instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Trained FOP Affinity Predictor**
   ```bash
   # Should exist at: /home/aaryan0302/FOP-Code/trained_models/best_model.ckpt
   ls trained_models/best_model.ckpt
   ```

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
cd /home/aaryan0302/FOP-Code/docker/gcdm

# Build image (~15 GB, takes ~10-15 minutes)
docker-compose build
```

This will:
- Create Ubuntu 22.04 + CUDA 11.8 environment
- Install Mambaforge and all dependencies
- Clone GCDM-SBDD repository
- Download pre-trained checkpoints (~500 MB)
- Set up Flask API for communication

### 2. Start the Container

```bash
# Start in background
docker-compose up -d

# Check logs
docker-compose logs -f

# Should see: "GCDM-SBDD Flask API Server" and "CUDA available: True"
```

### 3. Test the Setup

```bash
cd /home/aaryan0302/FOP-Code

# Test Docker GCDM client
python3 models/docker_gcdm_client.py

# Should show:
# ✓ GCDM API is healthy
# ✓ CUDA available: True
```

### 4. Generate Molecules!

```bash
# Generate molecules with affinity guidance
python3 generate_with_guidance_docker.py \
    --pdb data/structures/acvr1_wt_3mtf.pdb \
    --sequence "MTEYKLVVVGAGGVGKSALTIQLIQ..." \
    --resi-list A:1 A:2 A:3 A:4 A:5 \
    --n-samples 50 \
    --top-k 10 \
    --output-dir generated_molecules
```

## 📖 Usage

### Using Python API

```python
from models.docker_gcdm_client import DockerGCDMClient
from generate_with_guidance_docker import GuidedGCDMDockerGenerator

# Initialize generator (auto-starts container if needed)
generator = GuidedGCDMDockerGenerator(
    affinity_checkpoint="trained_models/best_model.ckpt",
    auto_start_container=True
)

# Generate and rank molecules
top_molecules = generator.generate_and_rank(
    pdb_file="data/structures/acvr1_wt_3mtf.pdb",
    protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQ...",
    resi_list=["A:1", "A:2", "A:3"],  # Define binding pocket
    n_samples=50,                      # Generate 50 molecules
    top_k=10                           # Return top 10 by affinity
)

# Save results
generator.save_results(
    molecules=top_molecules,
    output_dir="generated_molecules"
)
```

### Direct Docker GCDM Client

```python
from models.docker_gcdm_client import DockerGCDMClient

# Initialize client
client = DockerGCDMClient(api_url="http://localhost:5000")

# Generate molecules (without affinity guidance)
molecules = client.generate_molecules(
    pdb_file="data/structures/protein.pdb",
    resi_list=["A:1", "A:2", "A:3"],
    n_samples=20
)

# molecules is a list of dicts with: smiles, mol_weight, qed, logp, etc.
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Host Machine (ARM64 or x86_64)                              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ FOP-Code Project                                   │    │
│  │                                                     │    │
│  │  • Affinity Predictor (trained_models/)           │    │
│  │  • Data (data/)                                    │    │
│  │  • Python Scripts                                  │    │
│  │    - generate_with_guidance_docker.py             │    │
│  │    - models/docker_gcdm_client.py                 │    │
│  └────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            │ HTTP API (port 5000)            │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Docker Container (Linux x86_64 + CUDA)            │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ GCDM-SBDD                                    │ │    │
│  │  │  • Diffusion model (checkpoints/)           │ │    │
│  │  │  • PyTorch Geometric + CUDA                 │ │    │
│  │  │  • pytorch-scatter                           │ │    │
│  │  │  • Flask API (gcdm_api.py)                  │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │                                                     │    │
│  │  Mounted Volumes:                                  │    │
│  │    /workspace/fop-data        → ../../data         │    │
│  │    /workspace/fop-output      → ../../generated_*  │    │
│  │    /workspace/fop-models      → ../../trained_*    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Container Management

```bash
# Start container
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Restart container
docker-compose restart

# Check status
docker ps | grep gcdm

# Enter container shell
docker exec -it fop-gcdm-sbdd bash

# Inside container:
conda activate GCDM-SBDD
python test_imports.py
```

## 📁 Volume Mounts

The container has access to your local files via mounted volumes:

| Container Path | Host Path | Purpose |
|----------------|-----------|---------|
| `/workspace/fop-data` | `../../data` | PDB files, structures, ligands (read-only) |
| `/workspace/fop-output` | `../../generated_molecules` | Generated molecules output |
| `/workspace/fop-models` | `../../trained_models` | Affinity predictor checkpoints (read-only) |

## 🎛️ Configuration

### Available GCDM Checkpoints

The container includes these pre-trained models:

- `checkpoints/bindingmoad_ca_cond_gcpnet.ckpt` (recommended)
- `checkpoints/bindingmoad_ca_cond_egnn.ckpt`
- `checkpoints/bindingmoad_ca_joint_gcpnet.ckpt`
- `checkpoints/crossdock_ca_cond_gcpnet.ckpt`
- `checkpoints/crossdock_ca_cond_egnn.ckpt`

### Docker Compose Options

Edit `docker-compose.yml` to customize:

```yaml
# Use specific GPU
environment:
  - CUDA_VISIBLE_DEVICES=0  # or 1, 2, etc.

# Change API port
ports:
  - "5001:5000"  # Use port 5001 on host

# Allocate more memory
deploy:
  resources:
    limits:
      memory: 16G
```

## 🐛 Troubleshooting

### Container won't start
```bash
# Check Docker is running
docker ps

# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# View container logs
docker-compose logs
```

### API not responding
```bash
# Check if port is in use
lsof -i :5000

# Check container health
docker ps
# STATUS should be "healthy"

# View API logs
docker exec fop-gcdm-sbdd tail -f /workspace/GCDM-SBDD/api.log
```

### Generation fails
```bash
# Check PDB file path mapping
# Local: data/structures/protein.pdb
# Container: /workspace/fop-data/structures/protein.pdb

# Verify file exists in container
docker exec fop-gcdm-sbdd ls /workspace/fop-data/structures/
```

### Out of memory
```bash
# Reduce batch size or n_samples
# Monitor GPU memory
watch -n 1 nvidia-smi

# Or use CPU mode (slower)
# In generate call: device="cpu"
```

## 🔄 Workflow Integration

### Full Pipeline: Training → Generation → Evaluation

```bash
# 1. Train affinity predictor (on host)
python3 train_model.py --epochs 50

# 2. Start GCDM container
cd docker/gcdm
docker-compose up -d
cd ../..

# 3. Generate molecules with guidance
python3 generate_with_guidance_docker.py \
    --pdb data/structures/acvr1_wt_3mtf.pdb \
    --sequence "YOUR_PROTEIN_SEQUENCE" \
    --resi-list A:1 A:2 A:3 A:4 A:5 \
    --n-samples 100 \
    --top-k 20

# 4. Analyze results
python3 analyze_generated_molecules.py \
    generated_molecules/gcdm_guided_summary.csv
```

## 📊 Output Files

After generation, you'll have:

```
generated_molecules/
├── gcdm_guided_molecules.smi        # SMILES strings
├── gcdm_guided_predictions.json     # Full predictions with uncertainty
└── gcdm_guided_summary.csv          # Table: SMILES, MW, QED, pKd, k_off, etc.
```

## 🔬 Advanced Usage

### Custom Pocket Definition

```python
# Option 1: Residue list
generator.generate_and_rank(
    pdb_file="protein.pdb",
    resi_list=["A:10", "A:15", "A:20", "B:5"],  # Cross-chain pocket
    ...
)

# Option 2: Reference ligand
generator.generate_and_rank(
    pdb_file="protein_with_ligand.pdb",
    ref_ligand="A:403",  # Use ligand at chain A residue 403
    ...
)
```

### Batch Processing

```python
from pathlib import Path

generator = GuidedGCDMDockerGenerator(...)

for pdb_file in Path("data/structures").glob("*.pdb"):
    print(f"Processing {pdb_file.name}...")
    
    molecules = generator.generate_and_rank(
        pdb_file=pdb_file,
        protein_sequence=get_sequence(pdb_file),
        resi_list=get_pocket_residues(pdb_file),
        n_samples=50
    )
    
    generator.save_results(
        molecules=molecules,
        output_dir=f"results/{pdb_file.stem}"
    )
```

## 🚀 Performance

- **Generation time**: ~2-5 minutes for 50 molecules (GPU)
- **Container startup**: ~30-60 seconds (first time longer for downloads)
- **Disk space**: ~15 GB for image + checkpoints

## 📚 References

- GCDM-SBDD: https://github.com/BioinfoMachineLearning/GCDM-SBDD
- Paper: Morehead & Cheng (2024) "Geometry-complete diffusion for 3D molecule generation"
- Docker: https://docs.docker.com/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/

## ⚖️ License

- GCDM-SBDD: MIT License
- FOP-Code: (your license)

## 🙋 Support

Issues with:
- **Docker setup**: Check Docker/nvidia-docker installation
- **GCDM errors**: See GCDM-SBDD repository issues
- **Affinity predictor**: Check FOP-Code main README
