#!/bin/bash
# Script to start Docker on Vast.ai and launch GCDM container

echo "=== Starting Docker daemon ==="
# Kill any existing dockerd processes
pkill -9 dockerd 2>/dev/null
sleep 2

# Start dockerd in background with iptables disabled (for Vast.ai compatibility)
dockerd --iptables=false --ip-masq=false > /tmp/dockerd.log 2>&1 &
DOCKERD_PID=$!

echo "Waiting for Docker daemon to start (PID: $DOCKERD_PID)..."
# Wait up to 15 seconds for Docker to be ready
for i in {1..15}; do
    if docker ps >/dev/null 2>&1; then
        echo "✓ Docker daemon is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 1
done

# Verify Docker is working
if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: Docker daemon failed to start"
    echo "Last 20 lines of dockerd log:"
    tail -20 /tmp/dockerd.log
    exit 1
fi

echo ""
echo "=== Docker daemon started successfully ==="
docker version

echo ""
echo "=== Building and starting GCDM container ==="
cd /root/FOP-Code/docker/gcdm

# Build the image
echo "Building Docker image..."
docker compose build

# Start the container
echo "Starting container..."
docker compose up -d

# Wait for container to be ready
echo "Waiting for container to start..."
sleep 10

# Show status
echo ""
echo "=== Container Status ==="
docker compose ps
docker ps

echo ""
echo "=== Container Logs ==="
docker compose logs --tail=50

echo ""
echo "=== Done! ==="
