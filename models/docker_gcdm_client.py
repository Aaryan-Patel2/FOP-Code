"""
Docker-based GCDM Integration
Wrapper to communicate with GCDM-SBDD running in a Docker container
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import subprocess
import sys


class DockerGCDMClient:
    """
    Client for interacting with GCDM-SBDD running in Docker
    
    This class provides a Python interface to the GCDM diffusion model
    running in a Docker container, avoiding ARM64 compatibility issues.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:5000",
        checkpoint: str = "checkpoints/bindingmoad_ca_cond_gcpnet.ckpt",
        timeout: int = 300
    ):
        """
        Initialize Docker GCDM client
        
        Args:
            api_url: URL of the GCDM API server (default: http://localhost:5000)
            checkpoint: Path to model checkpoint within container
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.checkpoint = checkpoint
        self.timeout = timeout
        
        print("=" * 70)
        print("Docker-based GCDM Client")
        print("=" * 70)
        print(f"API URL: {self.api_url}")
        print(f"Checkpoint: {self.checkpoint}")
        
        # Check if container is running
        if not self._check_health():
            print("\n⚠ Warning: GCDM container not responding")
            print("   Start the container with: docker-compose up -d")
            print("   Or check status with: docker ps")
    
    def _check_health(self) -> bool:
        """Check if GCDM API is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ GCDM API is healthy")
                print(f"  CUDA available: {data.get('cuda_available', False)}")
                print(f"  Models loaded: {data.get('models_loaded', 0)}")
                return True
        except Exception as e:
            print(f"✗ Health check failed: {e}")
        return False
    
    def start_container(self, compose_file: Optional[str] = None):
        """
        Start the GCDM Docker container
        
        Args:
            compose_file: Path to docker-compose.yml (default: docker/gcdm/docker-compose.yml)
        """
        if compose_file is None:
            compose_file = str(Path(__file__).parent.parent / "docker" / "gcdm" / "docker-compose.yml")
        
        compose_dir = Path(compose_file).parent
        
        print(f"\n🐳 Starting GCDM container...")
        print(f"   Compose file: {compose_file}")
        
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=compose_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            print("✓ Container started successfully")
            
            # Wait for container to be ready
            print("   Waiting for API to be ready...", end="", flush=True)
            for i in range(30):
                time.sleep(2)
                if self._check_health():
                    print(" ✓")
                    return True
                print(".", end="", flush=True)
            
            print("\n⚠ Container started but API not responding")
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to start container: {e.stderr}")
            return False
        except FileNotFoundError:
            print("✗ docker-compose not found. Please install Docker Compose.")
            return False
    
    def stop_container(self, compose_file: Optional[str] = None):
        """Stop the GCDM Docker container"""
        if compose_file is None:
            compose_file = str(Path(__file__).parent.parent / "docker" / "gcdm" / "docker-compose.yml")
        
        compose_dir = Path(compose_file).parent
        
        print(f"\n🛑 Stopping GCDM container...")
        
        try:
            subprocess.run(
                ["docker-compose", "down"],
                cwd=compose_dir,
                check=True
            )
            print("✓ Container stopped")
        except Exception as e:
            print(f"✗ Failed to stop container: {e}")
    
    def generate_molecules(
        self,
        pdb_file: Union[str, Path],
        resi_list: Optional[List[str]] = None,
        ref_ligand: Optional[str] = None,
        n_samples: int = 20,
        sanitize: bool = True,
        fix_n_nodes: bool = False,
        device: str = "cuda"
    ) -> List[Dict]:
        """
        Generate molecules for a given pocket
        
        Args:
            pdb_file: Path to PDB file (will be mapped to container)
            resi_list: List of residue IDs defining pocket (e.g., ["A:1", "A:2"])
            ref_ligand: Reference ligand in format "chain:resi" (alternative to resi_list)
            n_samples: Number of molecules to generate
            sanitize: Whether to sanitize molecules
            fix_n_nodes: Fix number of atoms to match reference
            device: Device to use ('cuda' or 'cpu')
        
        Returns:
            List of molecule dictionaries with SMILES and properties
        """
        # Convert local path to container path
        pdb_file = Path(pdb_file)
        
        # Map local data directory to container directory
        # Assumes PDB is in data/ directory
        if 'data' in pdb_file.parts:
            idx = pdb_file.parts.index('data')
            container_path = Path('/workspace/fop-data') / Path(*pdb_file.parts[idx+1:])
        else:
            container_path = pdb_file
        
        # Prepare request
        request_data = {
            'checkpoint': self.checkpoint,
            'pdb_file': str(container_path),
            'n_samples': n_samples,
            'sanitize': sanitize,
            'fix_n_nodes': fix_n_nodes,
            'device': device
        }
        
        if resi_list:
            request_data['resi_list'] = resi_list
        elif ref_ligand:
            request_data['ref_ligand'] = ref_ligand
        else:
            raise ValueError("Either resi_list or ref_ligand must be provided")
        
        print(f"\n🔬 Generating {n_samples} molecules...")
        print(f"   PDB: {pdb_file.name}")
        print(f"   Pocket: {resi_list or ref_ligand}")
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                molecules = result.get('molecules', [])
                print(f"✓ Generated {len(molecules)} valid molecules")
                return molecules
            else:
                error = result.get('error', 'Unknown error')
                print(f"✗ Generation failed: {error}")
                if 'traceback' in result:
                    print(f"\nTraceback:\n{result['traceback']}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"✗ Request timed out after {self.timeout} seconds")
            return []
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            return []
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return []
    
    def list_available_checkpoints(self) -> List[str]:
        """List available model checkpoints in container"""
        try:
            response = requests.get(
                f"{self.api_url}/available_checkpoints",
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result.get('checkpoints', [])
        except Exception as e:
            print(f"Failed to list checkpoints: {e}")
            return []
    
    def molecules_to_rdkit(self, molecules: List[Dict]) -> List[Chem.Mol]:
        """
        Convert molecule dictionaries to RDKit Mol objects
        
        Args:
            molecules: List of molecule dicts with 'smiles' key
        
        Returns:
            List of RDKit Mol objects
        """
        rdkit_mols = []
        for mol_dict in molecules:
            smiles = mol_dict.get('smiles')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    rdkit_mols.append(mol)
        return rdkit_mols


def test_docker_gcdm():
    """Test function for Docker GCDM client"""
    print("Testing Docker GCDM Client")
    print("=" * 70)
    
    # Initialize client
    client = DockerGCDMClient()
    
    # Check if container is running
    if not client._check_health():
        print("\nAttempting to start container...")
        if not client.start_container():
            print("\n❌ Cannot start container. Please check Docker installation.")
            return
    
    # List available checkpoints
    print("\n📦 Available checkpoints:")
    checkpoints = client.list_available_checkpoints()
    for ckpt in checkpoints:
        print(f"   - {ckpt}")
    
    # Test generation (requires a PDB file)
    pdb_file = Path("data/structures/acvr1_wt_3mtf.pdbqt")
    if pdb_file.exists():
        print(f"\n🧪 Testing molecule generation...")
        molecules = client.generate_molecules(
            pdb_file=pdb_file,
            resi_list=["A:1", "A:2", "A:3"],  # Example residues
            n_samples=5
        )
        
        if molecules:
            print(f"\n✓ Successfully generated {len(molecules)} molecules")
            print("\nSample molecule:")
            print(f"  SMILES: {molecules[0]['smiles']}")
            print(f"  MW: {molecules[0]['mol_weight']:.2f}")
            print(f"  QED: {molecules[0]['qed']:.2f}")
    else:
        print(f"\n⚠ Test PDB not found: {pdb_file}")
        print("  Skipping generation test")


if __name__ == "__main__":
    test_docker_gcdm()
