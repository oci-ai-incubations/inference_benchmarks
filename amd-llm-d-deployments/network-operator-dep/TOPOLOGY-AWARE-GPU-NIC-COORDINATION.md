# Topology-Aware GPU–NIC Coordination on MI300x

## Problem

When a pod requests 4 GPUs + 4 SRIOV RDMA VFs, Kubernetes assigns them independently.
The AMD GPU device plugin picks 4 GPUs, and the SRIOV device plugin picks 4 VFs, with
zero coordination. This can result in GPUs on NUMA 1 paired with NICs on NUMA 0, forcing
every RDMA transfer to take a socket hop through the UPI/xGMI interconnect — killing
bandwidth and adding latency.

## Node Topology (BM.GPU.MI300X.8)

Each MI300x bare-metal node has a symmetric layout:

| NUMA 0 | NUMA 1 |
|--------|--------|
| GPU `0000:11:00.0` | GPU `0000:8b:00.0` |
| GPU `0000:2f:00.0` | GPU `0000:aa:00.0` |
| GPU `0000:46:00.0` | GPU `0000:c2:00.0` |
| GPU `0000:5d:00.0` | GPU `0000:da:00.0` |
| NIC `0000:0c:00.0` rdma0 | NIC `0000:86:00.0` rdma4 |
| NIC `0000:2a:00.0` rdma1 | NIC `0000:a5:00.0` rdma5 |
| NIC `0000:41:00.0` rdma2 | NIC `0000:bd:00.0` rdma6 |
| NIC `0000:58:00.0` rdma3 | NIC `0000:d5:00.0` rdma7 |

4 GPUs + 4 NICs per NUMA node.  A request for 4+4 fits entirely on one socket.

## Root Cause

The kubelet config (`/etc/kubernetes/kubelet-config.json`) has **no Topology Manager
policy** and **no CPU Manager policy** set, so the default `none` policy is active.
Both device plugins allocate independently with no NUMA awareness.

## Fix: Enable Kubelet Topology Manager

### What to change

Add these fields to `/etc/kubernetes/kubelet-config.json` on every GPU node:

```json
"topologyManagerPolicy": "restricted",
"topologyManagerScope": "container",
"topologyManagerPolicyOptions": {
  "prefer-closest-numa-nodes": "true"
},
"cpuManagerPolicy": "static"
```

**`prefer-closest-numa-nodes: true`** is critical. Without it (the default is `false`),
the Topology Manager does not prefer NUMA-local allocations when multiple NUMA sets of
equal width are available. The kubelet log will confirm:

```
topologyPolicyOptions={"PreferClosestNUMA":true, ...}
```

### Why `restricted`

| Policy | 4 GPU + 4 VF | 8 GPU + 8 VF |
|--------|--------------|--------------|
| `none` (current) | No alignment — random | No alignment — random |
| `best-effort` | Tries alignment, allows misalign | Allows everything |
| **`restricted`** | **Forces single-NUMA** | **Allows cross-NUMA (must, since 8+8 > one socket)** |
| `single-numa-node` | Forces single-NUMA | **Rejects pod** (can't fit on one socket) |

`restricted` is the right choice: it enforces NUMA alignment when possible (4+4), but
does not reject pods that legitimately need both sockets (8+8).

### How to apply

On each GPU node (via SSH or a privileged daemonset):

```bash
# 1. Back up
cp /etc/kubernetes/kubelet-config.json /etc/kubernetes/kubelet-config.json.bak

# 2. Patch — add topology manager + CPU manager fields
python3 -c "
import json
with open('/etc/kubernetes/kubelet-config.json') as f:
    cfg = json.load(f)
cfg['topologyManagerPolicy'] = 'restricted'
cfg['topologyManagerScope'] = 'container'
cfg['topologyManagerPolicyOptions'] = {'prefer-closest-numa-nodes': 'true'}
cfg['cpuManagerPolicy'] = 'static'
with open('/etc/kubernetes/kubelet-config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patched kubelet-config.json')
"

# 3. Remove stale CPU Manager state (required when switching to static policy)
rm -f /var/lib/kubelet/cpu_manager_state

# 4. Restart kubelet
systemctl restart kubelet
```

Or remotely through a debug pod / the sriov-network-config-daemon:

```bash
NODE=10.0.65.77
kubectl exec -n nvidia-network-operator \
  $(kubectl get pods -n nvidia-network-operator -l app=sriov-network-config-daemon \
    --field-selector spec.nodeName=$NODE -o name | head -1) -- \
  /bin/bash -c '
    cp /host/etc/kubernetes/kubelet-config.json /host/etc/kubernetes/kubelet-config.json.bak
    python3 -c "
import json
with open(\"/host/etc/kubernetes/kubelet-config.json\") as f:
    cfg = json.load(f)
cfg[\"topologyManagerPolicy\"] = \"restricted\"
cfg[\"topologyManagerScope\"] = \"container\"
cfg[\"topologyManagerPolicyOptions\"] = {\"prefer-closest-numa-nodes\": \"true\"}
cfg[\"cpuManagerPolicy\"] = \"static\"
with open(\"/host/etc/kubernetes/kubelet-config.json\", \"w\") as f:
    json.dump(cfg, f, indent=2)
print(\"Patched\")
"
    rm -f /host/var/lib/kubelet/cpu_manager_state
    chroot /host/ systemctl restart kubelet
  '
```

**Important:** Restarting kubelet will briefly make the node `NotReady`. Drain the node
first if it has running workloads.

### Verify

```bash
# 1. Check kubelet picked up the config
kubectl get node $NODE -o jsonpath='{.metadata.annotations}' | python3 -m json.tool | grep topology

# 2. Or check kubelet logs on the node
journalctl -u kubelet | grep -i "topology manager"
# Should show: "Topology Manager policy: restricted"

# 3. Deploy a test pod requesting 4 GPUs + 4 VFs
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: topology-test
  namespace: dk
  annotations:
    k8s.v1.cni.cncf.io/networks: |
      [
        {"name":"sriov-rdma-vf","namespace":"default","interface":"net1"},
      ]
spec:
  tolerations:
    - operator: Exists
  restartPolicy: Never
  terminationGracePeriodSeconds: 1
  nodeSelector:
    kubernetes.io/hostname: 10.0.65.77
  containers:
  - name: test
    image: docker.io/ubuntu:22.04
    command: ["sleep", "3600"]
    resources:
      limits:
        amd.com/gpu: 1
        nvidia.com/sriov-rdma-vf: 1
      requests:
        amd.com/gpu: 1
        nvidia.com/sriov-rdma-vf: 1
EOF

# 4. Check NUMA alignment
kubectl exec -n dk topology-test -- bash -c '
  echo "=== Assigned GPUs ==="
  for render in /dev/dri/renderD*; do
    minor=$(basename $render | sed "s/renderD//")
    pci_path=$(readlink -f /sys/class/drm/renderD${minor}/device 2>/dev/null)
    if [ -n "$pci_path" ]; then
      pci=$(basename $pci_path)
      numa=$(cat ${pci_path}/numa_node 2>/dev/null)
      echo "  $render -> PCI $pci -> NUMA $numa"
    fi
  done
  echo ""
  echo "=== Assigned NIC VFs ==="
  printenv "PCIDEVICE_NVIDIA_COM_SRIOV-RDMA-VF" | tr "," "\n" | while read vf; do
    numa=$(cat /sys/bus/pci/devices/$vf/numa_node 2>/dev/null)
    echo "  VF $vf -> NUMA $numa"
  done
'
```

**Expected result:** All 4 GPUs and all 4 VFs should report the same NUMA node.

## How It Works

1. Pod requests `amd.com/gpu: 4` + `nvidia.com/sriov-rdma-vf: 4`
2. Kubelet asks each device plugin for **topology hints**:
   - AMD GPU plugin: "I can give 4 from NUMA 0, or 4 from NUMA 1"
   - SRIOV plugin: "I can give 4 from NUMA 0, or 4 from NUMA 1"
3. Topology Manager **intersects** the hints: NUMA 0 or NUMA 1 both work
4. Topology Manager picks one (e.g., NUMA 0)
5. Both plugins allocate from that NUMA node
6. Result: GPU 0–3 + rdma0–3 (all NUMA 0), or GPU 4–7 + rdma4–7 (all NUMA 1)

## Rollback

If Topology Manager causes scheduling issues:

```bash
# Revert kubelet config
cp /etc/kubernetes/kubelet-config.json.bak /etc/kubernetes/kubelet-config.json
rm -f /var/lib/kubelet/cpu_manager_state
systemctl restart kubelet
```

## Notes

- Both device plugins already report topology (SRIOV confirmed via `ListAndWatch` logs;
  AMD GPU plugin reads `/sys/bus/pci/devices/<dev>/numa_node`)
- `cpuManagerPolicy: static` is required for Topology Manager to function properly
- The `kubeReserved.cpu: 637m` already set in the config satisfies the static CPU
  manager requirement of having reserved CPUs
- This change must be applied to every GPU node in the cluster
- OKE may reset kubelet config on node pool updates — consider automating via a
  DaemonSet or cloud-init userdata

Notes:
```
Topology Manager gets you same-NUMA (same-socket), but it does NOT guarantee the nearest PCIe-switch pair within that NUMA node. It only understands NUMA granularity — it has no concept of PCIe switch topology.
Your MI300x nodes almost certainly have this PCIe layout:
Socket 0                              Socket 1├─ PCIe switch 0: GPU 11 + NIC 0c    ├─ PCIe switch 4: GPU 8b + NIC 86├─ PCIe switch 1: GPU 2f + NIC 2a    ├─ PCIe switch 5: GPU aa + NIC a5├─ PCIe switch 2: GPU 46 + NIC 41    ├─ PCIe switch 6: GPU c2 + NIC bd├─ PCIe switch 3: GPU 5d + NIC 58    ├─ PCIe switch 7: GPU da + NIC d5
The Topology Manager eliminated the worst case (cross-socket hop), but within NUMA 0, GPU 46:00.0 talking to NIC 2a:00.1 still goes through the CPU's PCIe root complex instead of a direct PCIe switch. That's PXB-level (cross-bridge) instead of PIX-level (same-switch) — less latency than a socket hop, but not zero.
Does this matter in practice? For your workloads (LLM inference with RDMA KV-cache transfer), the socket hop was the big performance killer. Same-NUMA-different-switch is significantly better than cross-socket, even if not as good as same-switch. The difference between PIX and PXB within the same NUMA is usually single-digit microseconds of latency and modest bandwidth differences.
For truly optimal same-switch pairing, Kubernetes Topology Manager can't help — it doesn't understand PCIe switch topology. The options would be:
DRA (Dynamic Resource Allocation) — Your cluster already has DynamicResourceAllocation: true in the kubelet feature gates. A DRA driver could understand the full PCIe topology and allocate GPU+NIC pairs at PCIe-switch granularity. This is the "right" long-term solution but requires writing/deploying a custom DRA driver.
Application-level mapping — If your workload (vLLM) uses PCIDEVICE_NVIDIA_COM_SRIOV-RDMA-VF to map NICs to GPUs, it could sort/match by PCI bus proximity at runtime rather than relying on assignment order.
Bottom line: You went from random cross-socket assignments to guaranteed same-NUMA. That's the big win. The remaining same-NUMA-different-switch gap is real but much smaller.
```