#!/bin/bash
# Investigation script for VF DOWN state on node 10.0.78.230
# Pod: ms-disagg-llm-d-modelservice-prefill-65b4865549-6cns (ns: vinccave-run-4k256-llama3-disagg-2p-tp1-1d-tp2)

NS="${1:-vinccave-run-4k256-llama3-disagg-2p-tp1-1d-tp2}"
POD="${2:-}"  # Auto-detect if empty
NODE_IP="${3:-10.0.78.230}"

# Resolve node name (SriovNetworkNodeState uses node name, not IP)
NODE=$(kubectl get nodes -o json 2>/dev/null | jq -r --arg ip "$NODE_IP" '.items[] | select(.status.addresses[]? | select(.type=="InternalIP" and .address==$ip)) | .metadata.name' 2>/dev/null | head -1)
NODE=${NODE:-$NODE_IP}

# Auto-detect pod on node if not specified
if [ -z "$POD" ]; then
  POD=$(kubectl get pods -n "$NS" -o json 2>/dev/null | jq -r --arg node "$NODE" --arg n "$NODE_IP" '.items[] | select(.spec.nodeName==$node or .spec.nodeName==$n) | select(.metadata.name | test("prefill|sriov|modelservice")) | .metadata.name' 2>/dev/null | head -1)
fi
[ -z "$POD" ] && POD="ms-disagg-llm-d-modelservice-prefill-65b4865549-6cnsg"

echo "=== 1. Pod status and assigned resources (ns=$NS pod=$POD) ==="
if ! kubectl get pod -n "$NS" "$POD" -o wide 2>/dev/null; then
  echo "ERROR: Pod not found. List pods in namespace:"
  kubectl get pods -n "$NS" -o wide 2>/dev/null || true
  echo ""
  echo "Usage: $0 [namespace] [pod-name] [node-ip]"
  exit 1
fi

echo ""
echo "=== 1b. Pod events (check for FailedCreatePodSandBox / nv-ipam errors) ==="
kubectl get events -n "$NS" --field-selector involvedObject.name="$POD" --sort-by='.lastTimestamp' 2>/dev/null | tail -15

echo ""
echo "=== 2. Pod network annotations (k8s.v1.cni.cncf.io/networks) ==="
kubectl get pod -n "$NS" "$POD" -o jsonpath='{.metadata.annotations.k8s\.v1\.cni\.cncf\.io/networks}' 2>/dev/null | jq . 2>/dev/null || \
kubectl get pod -n "$NS" "$POD" -o jsonpath='{.metadata.annotations.k8s\.v1\.cni\.cncf\.io/networks}' 2>/dev/null
echo ""

echo "=== 3. Network attachment status (k8s.v1.cni.cncf.io/networks-status) ==="
kubectl get pod -n "$NS" "$POD" -o jsonpath='{.metadata.annotations.k8s\.v1\.cni\.cncf\.io/networks-status}' 2>/dev/null | jq . 2>/dev/null || \
kubectl get pod -n "$NS" "$POD" -o jsonpath='{.metadata.annotations.k8s\.v1\.cni\.cncf\.io/networks-status}' 2>/dev/null
echo ""

echo "=== 4. Container names (for exec) ==="
kubectl get pod -n "$NS" "$POD" -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}' 2>/dev/null

echo ""
echo "=== 5. SR-IOV resource request/limit ==="
kubectl get pod -n "$NS" "$POD" -o jsonpath='{range .spec.containers[*]}Container: {.name}{"\n"}  limits: {.resources.limits}{"\n"}  requests: {.resources.requests}{"\n"}{end}' 2>/dev/null | head -20

echo ""
echo "=== 6. PCIDEVICE env (assigned VF BDFs) - run inside pod ==="
CONTAINER=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.spec.containers[0].name}' 2>/dev/null)
READY=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
if [ "$READY" != "Running" ]; then
  echo "(Skipping exec - pod not Running yet, phase=$READY. Fix network attachment first.)"
else
  kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- printenv 2>/dev/null | grep -i pcidevice || echo "(exec may fail if pod has security restrictions)"
fi

echo ""
echo "=== 7. RDMA link state INSIDE pod (requires rdma-core or similar) ==="
if [ "$READY" = "Running" ]; then
  kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- rdma link 2>/dev/null || echo "(rdma link not available in container)"
else
  echo "(Skipped - pod not Running)"
fi

echo ""
echo "=== 8. IP link INSIDE pod ==="
if [ "$READY" = "Running" ]; then
  kubectl exec -n "$NS" "$POD" -c "$CONTAINER" -- ip link 2>/dev/null || echo "(ip not available)"
else
  echo "(Skipped - pod not Running)"
fi

echo ""
echo "=== 9. Node $NODE - VF capacity/allocatable ==="
kubectl get node "$NODE" -o jsonpath='Capacity: {.status.capacity.nvidia\.com/sriov-rdma-vf}{"\n"}Allocatable: {.status.allocatable.nvidia\.com/sriov-rdma-vf}{"\n"}' 2>/dev/null

echo ""
echo "=== 10. SriovNetworkNodeState for node ==="
kubectl get sriovnetworknodestate "$NODE" -n nvidia-network-operator -o yaml 2>/dev/null | head -80
