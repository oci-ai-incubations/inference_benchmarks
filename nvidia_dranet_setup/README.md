# NVIDIA DRA + dranet GPU-NIC Co-allocation Setup on OKE

GPU-NIC co-allocation using Kubernetes Dynamic Resource Allocation (DRA) to guarantee
that NVIDIA GPUs and their nearest RDMA NICs are always assigned together, ensuring
peak RDMA performance without PCIe root complex traversal.

## Problem

Standard Kubernetes scheduling (SR-IOV device plugin, topology-aware scheduling) cannot
guarantee that a GPU and NIC share the same PCIe root complex. At best, NUMA-restricted
scheduling places them on the same NUMA node, but traffic may still traverse the PCIe
root complex -- a major performance penalty for RDMA workloads.

## Solution

Two DRA drivers publish ResourceSlices with `resource.kubernetes.io/pcieRoot` attributes:

- **dranet** (`dra.net`) -- discovers RDMA NICs and their PCIe topology
- **NVIDIA k8s-dra-driver-gpu** (`gpu.nvidia.com`) -- discovers GPUs and their PCIe topology

A `ResourceClaimTemplate` with `matchAttribute: "resource.kubernetes.io/pcieRoot"` constraint
forces the scheduler to co-allocate only devices sharing the same PCIe root.

## Cluster Environment

- **Platform**: Oracle Kubernetes Engine (OKE)
- **Kubernetes**: v1.34.1
- **GPU Nodes**: 9x `BM.GPU.H100.8` (72x NVIDIA H100 80GB HBM3 total)
- **NICs**: 16x Mellanox ConnectX-7 per GPU node (2 per GPU, MTU 4220)
- **RDMA Cluster**: All GPU nodes in same RDMA cluster (`ezbiodkjviq`)
- **GPU Driver**: CUDA 580.126.20 (pre-installed on host)
- **GPU Operator**: v25.10.0
- **Container Runtime**: CRI-O 1.34.0

### PCIe Topology (BM.GPU.H100.8 / H100T variant)

Each GPU shares a PCIe root with exactly 2 CX-7 RDMA NICs:

| PCIe Root    | NUMA | GPU PCI Bus | NIC 1 PCI Bus | NIC 2 PCI Bus |
|--------------|------|-------------|---------------|---------------|
| pci0000:07   | 0    | 0000:0f:00.0 | 0000:0c:00.0 | 0000:0c:00.1 |
| pci0000:25   | 0    | 0000:2d:00.0 | 0000:2a:00.0 | 0000:2a:00.1 |
| pci0000:3c   | 0    | 0000:44:00.0 | 0000:41:00.0 | 0000:41:00.1 |
| pci0000:53   | 0    | 0000:5b:00.0 | 0000:58:00.0 | 0000:58:00.1 |
| pci0000:81   | 1    | 0000:89:00.0 | 0000:86:00.0 | 0000:86:00.1 |
| pci0000:a0   | 1    | 0000:a8:00.0 | 0000:a5:00.0 | 0000:a5:00.1 |
| pci0000:b8   | 1    | 0000:c0:00.0 | 0000:bd:00.0 | 0000:bd:00.1 |
| pci0000:d0   | 1    | 0000:d8:00.0 | 0000:d5:00.0 | 0000:d5:00.1 |

Note: OKE labels these nodes as `BM.GPU.H100.8` but the PCIe layout matches the H100T
variant (PCI addresses: 0c, 2a, 41, 58, 86, a5, bd, d5). dranet discovers this dynamically.

## Setup Steps

### Step 1: Deploy dranet (DONE)

dranet is deployed cluster-wide as a DaemonSet. It discovers all network interfaces
and publishes ResourceSlices with PCIe topology attributes.

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/dranet/main/install.yaml
```

Verify ResourceSlices are published:
```bash
kubectl get resourceslices -l resource.k8s.io/driver=dra.net
```

Each GPU node should show 16 ConnectX-7 devices with `dra.net/rdma: true` and
`resource.kubernetes.io/pcieRoot` attributes.

### Step 2: Label test nodes for DRA (DONE)

Selected 2 GPU nodes for phased DRA rollout (different fault domains):

```bash
kubectl label node 10.0.68.50 nvidia.com/dra-kubelet-plugin=true
kubectl label node 10.0.70.13 nvidia.com/dra-kubelet-plugin=true
```

### Step 3: Disable legacy device plugin on test nodes (DONE)

The NVIDIA DRA driver and legacy device plugin cannot coexist on the same node
(double-counting GPUs, scheduling conflicts). Disable the device plugin on DRA nodes:

```bash
kubectl label node 10.0.68.50 nvidia.com/gpu.deploy.device-plugin=false --overwrite
kubectl label node 10.0.70.13 nvidia.com/gpu.deploy.device-plugin=false --overwrite
```

This causes the GPU Operator to remove the device plugin pods from these nodes.
The remaining 7 GPU nodes continue using the legacy device plugin.

### Step 4: Deploy NVIDIA k8s-dra-driver-gpu (DONE)

Install via Helm targeting only labeled nodes:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update nvidia

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.12.0" \
  --create-namespace \
  --namespace nvidia-dra-driver-gpu \
  -f values.yaml
```

**values.yaml:**
```yaml
nvidiaDriverRoot: /
gpuResourcesEnabledOverride: true
logVerbosity: "4"

resources:
  gpus:
    enabled: true
  computeDomains:
    enabled: false

kubeletPlugin:
  nodeSelector:
    nvidia.com/dra-kubelet-plugin: "true"
  tolerations:
  - operator: Exists
    effect: NoSchedule
```

Key decisions:
- `nvidiaDriverRoot: /` -- GPU driver is pre-installed on host (not via GPU Operator driver container)
- `gpuResourcesEnabledOverride: true` -- GPU allocation is experimental; must be explicitly enabled
- `computeDomains.enabled: false` -- not needed (GB200/MNNVL); avoids controller scheduling issues on managed OKE (no schedulable control-plane nodes)
- `kubeletPlugin.nodeSelector` -- restricts DRA driver to labeled test nodes only

### Step 5: Validate pcieRoot alignment (DONE)

Confirmed that both drivers publish matching `resource.kubernetes.io/pcieRoot` values
for co-located GPU and NIC devices on node `10.0.68.50`:

```bash
# GPU ResourceSlices
kubectl get resourceslice 10.0.68.50-gpu.nvidia.com-qnrw7 -o json | \
  jq '.spec.devices[] | {name, pcieRoot: .attributes["resource.kubernetes.io/pcieRoot"].string}'

# NIC ResourceSlices (RDMA CX-7 only)
kubectl get resourceslice 10.0.68.50-dra.net-ls7fc -o json | \
  jq '.spec.devices[] | select(.attributes["dra.net/rdma"].bool == true) |
    {name, pcieRoot: .attributes["resource.kubernetes.io/pcieRoot"].string}'
```

All 8 PCIe roots match perfectly: each GPU shares a root with exactly 2 CX-7 NICs.

### Step 6: Create DeviceClasses and ResourceClaimTemplates (DONE)

Created DeviceClass for dranet (see `deviceclass-dranet.yaml`):
```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: dranet
spec:
  selectors:
  - cel:
      expression: device.driver == "dra.net"
```

The `gpu.nvidia.com` DeviceClass is auto-created by the NVIDIA DRA driver Helm chart.

Created ResourceClaimTemplate for co-allocated GPU + NIC (see `resourceclaim-gpu-nic-aligned.yaml`):
```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: gpu-nic-aligned
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly:
          deviceClassName: gpu.nvidia.com
          count: 1
      - name: nic
        exactly:
          deviceClassName: dranet
          count: 2
          selectors:
          - cel:
              expression: device.attributes["dra.net"].rdma == true
      constraints:
      - matchAttribute: "resource.kubernetes.io/pcieRoot"
```

Apply:
```bash
kubectl apply -f deviceclass-dranet.yaml
kubectl apply -f resourceclaim-gpu-nic-aligned.yaml -n kube-system
```

Note: ResourceClaimTemplate must be in the same namespace as the pod that references it.
Deployed to `kube-system` to bypass the Kueue validating webhook which intercepts all
pods in other namespaces.

### Step 7: Test co-allocation with a pod (DONE)

Deployed test pod (see `test-co-allocation.yaml`):
```bash
kubectl apply -f test-co-allocation.yaml
```

**Result: co-allocation confirmed.** The scheduler allocated all devices on the same PCIe root:

| Device | Driver | PCI Bus | PCIe Root |
|--------|--------|---------|-----------|
| gpu-2 | gpu.nvidia.com | 0000:44:00.0 | pci0000:3c |
| pci-0000-41-00-0 (rdma4) | dra.net | 0000:41:00.0 | pci0000:3c |
| pci-0000-41-00-1 (rdma5) | dra.net | 0000:41:00.1 | pci0000:3c |

Verified via:
```bash
kubectl -n kube-system get resourceclaim <claim-name> -o json | jq '.status.allocation.devices'
```

The GPU at PCI bus 0000:44:00.0 and both CX-7 NICs at 0000:41:00.0 / 0000:41:00.1 all
share PCIe root `pci0000:3c`, guaranteeing zero PCIe root complex traversal for RDMA.

### Step 8: NCCL / RDMA bandwidth validation (DONE)

Ran 2-node NCCL all_reduce bandwidth test with co-allocated GPU+NIC pairs (see `nccl-test.yaml`).
Each pod got 1 GPU + 1 RDMA NIC on the same PCIe root, using PyTorch distributed with NCCL backend.

```bash
kubectl apply -f nccl-test.yaml
# Wait for completion, then:
kubectl -n kube-system logs -l job-name=nccl-test --prefix
```

**Key NCCL debug output confirmed GPUDirect RDMA is active:**
```
NET/IB : Using [0]mlx5_9:1/RoCE [RO]; OOB eth0
DMA-BUF is available on GPU device 0
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IBext_v8/0/GDRDMA
Channel 01/0 : 0[0] -> 1[0] [send] via NET/IBext_v8/0/GDRDMA
```

NCCL used `GDRDMA` (GPUDirect RDMA) -- data flows directly between GPU memory and the
NIC via PCIe without CPU involvement, enabled by the co-allocation on the same PCIe root.

**Bandwidth results (2 nodes, 1 H100 + 1 CX-7 per node, all_reduce):**

| Size | Time (us) | AlgoBW (GB/s) | BusBW (GB/s) |
|------|-----------|---------------|--------------|
| 8 | 21.7 | 0.00 | 0.00 |
| 256 | 22.6 | 0.01 | 0.01 |
| 8K | 25.5 | 0.32 | 0.32 |
| 256K | 70.5 | 3.72 | 3.72 |
| 1M | 86.5 | 12.12 | 12.12 |
| 4M | 239.9 | 17.48 | 17.48 |
| 16M | 877.5 | 19.12 | 19.12 |
| 64M | 3446.9 | 19.47 | 19.47 |
| 256M | 12949.3 | 20.73 | 20.73 |
| 512M | 24687.4 | 21.75 | 21.75 |
| 1G | 48418.9 | 22.18 | 22.18 |

**Analysis:**
- Peak BusBW: **22.18 GB/s** at 1GB message size
- CX-7 single-port line rate: 200 Gb/s = 25 GB/s
- Achieving **~89% of theoretical line rate** at large message sizes
- The test used only 1 of 2 CX-7 ports per GPU; using both ports (count: 2 in the claim)
  and NCCL's multi-rail support could approach 50 GB/s per GPU
- NCCL latency at small sizes (~22 us) is consistent with direct PCIe-attached RDMA
  (no root complex traversal overhead)

### Comparison: DRA co-allocated test vs full-node NCCL test

A separate 72-GPU NCCL test ran across all 9 nodes using the legacy approach (all 8 GPUs +
all 16 NICs per node, hostNetwork-style, via `nccl-test-launcher`):

| Size | AlgBW (GB/s) | BusBW (GB/s) |
|------|-------------|--------------|
| 8M | 25.75 | 50.77 |
| 64M | 81.24 | 160.21 |
| 256M | 109.55 | 216.06 |
| 512M | 141.26 | 278.59 |
| 1G | 148.76 | 293.38 |
| 4G | 161.08 | **317.69** |

Peak BusBW: **317.76 GB/s** (16 ports x 25 GB/s = 400 GB/s theoretical, ~79% efficiency).
Per-port efficiency is comparable to our DRA test (89% single-port vs 79% aggregate across
16 ports and 9 nodes). The overhead at scale is expected for 72-GPU all_reduce coordination.

The full-node test benefits from NCCL's internal topology detection when all devices are
available. The DRA approach becomes critical for **partial allocation** -- when a pod gets
1-2 GPUs instead of all 8, the scheduler would otherwise have no guarantee which NICs to
pair, potentially assigning cross-PCIe-root NICs and destroying RDMA performance.

## Known Issues and Caveats

1. **NVIDIA GPU allocation is experimental** -- the GPU kubelet plugin in k8s-dra-driver-gpu
   is not officially supported yet (only ComputeDomains are GA). Enable at your own risk.

2. **k8s v1.34.2 recommended** -- a pcieRoot crash bug on certain VM types was fixed in
   v1.34.2 (Issue #575). Current cluster is on v1.34.1. Bare-metal nodes appear unaffected
   but upgrade is recommended.

3. **No simultaneous DRA + device plugin per node** -- GPUs would be double-counted.
   Use node-label segregation for phased rollout.

4. **OKE managed control plane** -- the DRA controller (for ComputeDomains) requires
   scheduling on control-plane nodes which are not accessible on OKE. Disabled via
   `computeDomains.enabled: false`.

5. **OCI shape label mismatch** -- nodes are labeled `BM.GPU.H100.8` but have the H100T
   PCIe layout. dranet handles this correctly via dynamic discovery (no static PCI maps).

## References

- [dranet project](https://github.com/kubernetes-sigs/dranet)
- [dranet + NVIDIA integration guide](https://dranet.dev/docs/user/nvidia-dranet/)
- [NVIDIA k8s-dra-driver-gpu](https://github.com/NVIDIA/k8s-dra-driver-gpu)
- [NVIDIA DRA driver installation wiki](https://github.com/NVIDIA/k8s-dra-driver-gpu/wiki/Installation)
- [OCI HPC OKE SR-IOV NIC policies](https://github.com/oracle-quickstart/oci-hpc-oke/blob/vf/manifests/sriov-network-node-policy.yaml)
- [Kubernetes DRA documentation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
