# Setup on nics as resources on MI300x

## Prerequisites: Helm operators

https://github.com/oracle-quickstart/oci-hpc-oke/tree/vf

1. Install the **nvidia** network operator (Mellanox NICs on MI300X):
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm upgrade -i network-operator nvidia/network-operator \
  -n nvidia-network-operator \
  --create-namespace \
  --version v25.10.0 \
  -f network-operator-values.yaml \
  --wait
```

The `network-operator-values.yaml` includes a toleration for `amd.com/gpu=present:NoSchedule` so NFD worker and SRIOV components can schedule on MI300x nodes (which typically have this taint).

---

## 1. Apply all resources

### Step 1a: Identify and label the target node

```bash
# List all nodes and their labels; find the one that shows <none> for sriov.oracle.com/target
kubectl describe nodes | grep -A 100 "Labels:" | grep -E "Name:|sriov.oracle.com/target|Labels:"

# Or list nodes with sriov label status:
kubectl get nodes -o custom-columns='NAME:.metadata.name,SRIOV-TARGET:.metadata.labels.sriov\.oracle\.com/target'
```

Identify the node that shows `<none>` for `sriov.oracle.com/target` (i.e., your target GPU/bare-metal node that does not yet have the label).

```bash
# Label the target node (replace <NODE_NAME> with the node name)
kubectl label node <NODE_NAME> sriov.oracle.com/target=true
```

### Step 1b: Apply YAMLs in order

All YAMLs use `sriov.oracle.com/target: "true"` in their nodeSelectors. Apply in this order:

```bash
kubectl apply -f nic_policy.yaml
kubectl apply -f ippool.yaml
kubectl apply -f bm.gpu.mi300x-sriov-policy.yaml
kubectl apply -f mi300x-network-pool-config.yaml   # or: sriov-network-pool-config-percentage.yaml
kubectl apply -f sriov.yaml
kubectl apply -f vfconfig.yaml
```

When this is completed, it may take a few minutes for the nodes to sync. Run this command:
```bash
kubectl get nodes -l 'amd.com/gpu=true' --sort-by=.status.capacity."amd\.com/gpu" -o=custom-columns='NODE:metadata.name,GPUs:status.capacity.amd\.com/gpu,RDMA-VFs:status.capacity.nvidia\.com/sriov-rdma-vf'

NODE          GPUs   RDMA-VFs
10.0.65.77    8      8
10.0.66.42    8      8
10.0.67.106   8      8
10.0.69.254   8      8
10.0.73.241   8      8
10.0.73.254   8      8
10.0.73.3     8      8
10.0.74.185   8      8
10.0.78.230   8      8
```

---

## 2. Delete all resources

Teardown in reverse dependency order:

```bash
kubectl delete -f vfconfig.yaml
kubectl delete -f sriov.yaml
kubectl delete -f mi300x-network-pool-config.yaml
kubectl delete -f sriov-network-pool-config-percentage.yaml   # only if you applied it
kubectl delete -f bm.gpu.mi300x-sriov-policy.yaml
kubectl delete -f ippool.yaml
kubectl delete -f nic_policy.yaml
```

### Optional: Remove node label after teardown

```bash
kubectl label node <NODE_NAME> sriov.oracle.com/target-
```

### Uninstall Helm operators

After the `kubectl delete` teardown, uninstall the Helm releases and optionally remove their namespaces:

```bash
# Uninstall NVIDIA network operator
helm uninstall network-operator -n nvidia-network-operator

# Optional: Remove the namespaces if they are empty
kubectl delete namespace nvidia-network-operator
```

**Note:** If namespaces are stuck in `Terminating`, check for remaining CRs, webhooks, or finalizers (e.g. `kubectl get all -n nvidia-network-operator`) and clean them before retrying.

---

## File reference

| # | File | Purpose |
|---|------|---------|
| 0 | network-operator-values.yaml | Helm values for NVIDIA network operator (NFD + SRIOV tolerations for amd.com/gpu) |
| 1 | nic_policy.yaml | Base Mellanox/NV-IPAM policy |
| 2 | ippool.yaml | NV-IPAM IP pool for SRIOV networks |
| 3 | bm.gpu.mi300x-sriov-policy.yaml | SRIOV node policy (defines VFs) |
| 4 | mi300x-network-pool-config.yaml | Pool config for rollout |
| 5 | sriov.yaml | SRIOV network using the pool and config |
| 6 | vfconfig.yaml | DaemonSet for VF configuration |

## Notes, debug

###  Debug / fix in ROCSHMEM
On each node, run this to force traffic class to be 41 (prio0 -> RDMA). See [main readme](../README.md)
```bash
echo 41 > /sys/class/infiniband/mlx5_0/tc/1/traffic_class
```

In container:
```bash
ARG GFX_COMPILATION_ARCH="gfx942"
ARG ROCSHMEM_BRANCH="feature/filter-nics"
RUN --mount=type=cache,target=/root/.ccache \
    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && \
    cd rocm-systems && \
    git remote add abouteiller https://github.com/abouteiller/rocm-systems.git && \
    git fetch abouteiller && \
    git sparse-checkout set --cone projects/rocshmem && \
    git checkout ${ROCSHMEM_BRANCH} && \
    git  -c user.name="Builder" -c user.email="build@local.com" merge --no-edit abouteiller/bugfix/traffic-class && \
    cd projects/rocshmem && \
    mkdir -p rocshmem-build && \
    cd rocshmem-build && \
    ../scripts/build_configs/all_backends \
        -DUSE_EXTERNAL_MPI=OFF \
        -DGPU_TARGETS=$GFX_COMPILATION_ARCH

# Environment variables used in app:
    export ROCSHMEM_HEAP_SIZE=8589934592
    export ROCSHMEM_MAX_NUM_CONTEXTS=64
    export ROCSHMEM_BACKEND=gda
    export ROCSHMEM_GDA_TRAFFIC_CLASS=41
    export ROCSHMEM_HCA_LIST=^mlx5_1,mlx5_6
```