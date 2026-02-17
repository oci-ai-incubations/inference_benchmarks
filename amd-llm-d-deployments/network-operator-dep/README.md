# Setup on nics as resources on MI300x

## Prerequisites: Helm operators

https://github.com/oracle-quickstart/oci-hpc-oke/tree/vf

# Debug / fix in ROCSHMEM
On each node, run this
```bash
echo 41 > /sys/class/infiniband/mlx5_0/tc/1/traffic_class
```

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