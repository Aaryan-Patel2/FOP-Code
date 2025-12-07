#!/bin/bash
# Build GCDM Docker image with platform specification
# This forces x86_64 architecture even on ARM64 hosts

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "Building GCDM Docker Image"
echo "========================================"
echo ""
echo "Platform: linux/amd64 (x86_64)"
echo "Note: This may be slow on ARM64 hosts due to emulation"
echo ""

# Build with platform flag
docker build --platform linux/amd64 -t gcdm-sbdd:latest .

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "To run the container:"
echo "  docker run --platform linux/amd64 -p 5000:5000 gcdm-sbdd:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d"
