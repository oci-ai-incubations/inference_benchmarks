# Troubleshooting Guide: SRIOV VF Creation Issue

## Problem Statement

Node `10.0.65.77` was showing `nvidia.com/sriov-rdma-vf: 0` in allocatable resources, preventing pods from requesting SRIOV VF resources. The `sriov-network-config-daemon` pod was reporting that a reboot was required even after the node had been rebooted.

## Initial Investigation

### 1. Node Status Check
```bash
kubectl describe node 10.0.65.77
```
**Findings:**
- Node annotation: `sriovnetwork.openshift.io/state: Reboot_Required`
- Capacity showed `nvidia.com/sriov-rdma-vf: 0`
- Allocatable showed `nvidia.com/sriov-rdma-vf: 0`
- Node was unschedulable: `node.kubernetes.io/unschedulable:NoSchedule`

### 2. Daemon Pod Logs Analysis
```bash
kubectl logs sriov-network-config-daemon-cp8wn -n nvidia-network-operator --all-containers=true
```
**Key Findings:**
- Daemon was detecting kernel argument changes requiring reboot:
  - Adding: `iommu=pt`
  - Adding: `ib_core.netns_mode=0`
  - Removing: `ib_core.netns_mode=1`
  - Removing: `pci=realloc`
  - Removing: `intel_iommu=on`
- Log message: `generic-plugin needRebootNode(): need reboot for updating kernel arguments`
- Daemon was continuously detecting these changes in a loop

### 3. Kernel Command Line Verification
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- cat /proc/cmdline
```
**Findings:**
- Running kernel had: `iommu=pt`
- Running kernel did **NOT** have: `ib_core.netns_mode=0`

### 4. GRUB Configuration Analysis
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- cat /host/etc/default/grub
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- cat /host/etc/default/grub.d/70-oci-hpc.cfg
```
**Critical Discovery:**
- `/etc/default/grub` had `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt ib_core.netns_mode=0"`
- `/etc/default/grub.d/70-oci-hpc.cfg` had `GRUB_CMDLINE_LINUX="... iommu=pt ..."` **without** `ib_core.netns_mode=0`
- The OCI HPC config file (`70-oci-hpc.cfg`) sets `GRUB_CMDLINE_LINUX`, which takes precedence over `GRUB_CMDLINE_LINUX_DEFAULT`
- The actual GRUB config (`/boot/grub/grub.cfg`) did not include `ib_core.netns_mode=0` in the kernel command line

### 5. VF Status Check
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- cat /host/sys/bus/pci/devices/0000:0c:00.0/sriov_numvfs
```
**Findings:**
- All PCI devices showed `sriov_numvfs = 0` (no VFs created)

### 6. SriovNetworkNodeState Status
```bash
kubectl get sriovnetworknodestate 10.0.65.77 -n nvidia-network-operator -o yaml
```
**Findings:**
- Annotation: `sriovnetwork.openshift.io/desired-state: Reboot_Required`
- Annotation: `sriovnetwork.openshift.io/current-state: Idle`
- Status: `syncStatus: InProgress`
- No VFs reported in status

## Root Causes Identified

### Cause 1: Kernel Argument Mismatch
The daemon was modifying `/etc/default/grub` (specifically `GRUB_CMDLINE_LINUX_DEFAULT`), but the actual kernel command line comes from `/etc/default/grub.d/70-oci-hpc.cfg` (which sets `GRUB_CMDLINE_LINUX`). The OCI HPC config file was missing `ib_core.netns_mode=0`, so even though GRUB had it configured, the kernel wasn't booted with it.

### Cause 2: Stale Reboot Required Annotation
The `Reboot_Required` annotation on both the node and SriovNetworkNodeState was preventing the daemon from proceeding with VF creation, even though the kernel arguments were eventually fixed.

## Solution Applied

### Step 1: Fix Kernel Arguments in OCI HPC Config
Added `ib_core.netns_mode=0` to the OCI HPC GRUB configuration file:

```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- /bin/bash -c \
  "sed -i 's/GRUB_CMDLINE_LINUX=\"\$GRUB_CMDLINE_LINUX processor.max_cstate=1 mce=ignore_ce skew_tick=1 iommu=pt/GRUB_CMDLINE_LINUX=\"\$GRUB_CMDLINE_LINUX processor.max_cstate=1 mce=ignore_ce skew_tick=1 iommu=pt ib_core.netns_mode=0/' /host/etc/default/grub.d/70-oci-hpc.cfg"
```

**Result:** The file now contains:
```
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX processor.max_cstate=1 mce=ignore_ce skew_tick=1 iommu=pt ib_core.netns_mode=0 numa_balancing=disable ..."
```

### Step 2: Regenerate GRUB Configuration
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- /bin/bash -c \
  "chroot /host/ update-grub"
```

**Verification:**
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- \
  grep -E "ib_core.netns_mode" /host/boot/grub/grub.cfg
```
Confirmed that `ib_core.netns_mode=0` is now in the GRUB config kernel command line.

### Step 3: Clear Stale Reboot Required Annotations
```bash
# Clear node annotation
kubectl annotate node 10.0.65.77 sriovnetwork.openshift.io/state- --overwrite

# Clear SriovNetworkNodeState annotation
kubectl annotate sriovnetworknodestate 10.0.65.77 -n nvidia-network-operator \
  sriovnetwork.openshift.io/desired-state- --overwrite
```

**Rationale:** These annotations were blocking VF creation. After clearing them, the daemon immediately began creating VFs.

## Verification Steps

### 1. Verify VFs Created on Host
```bash
kubectl exec -n nvidia-network-operator sriov-network-config-daemon-cp8wn -- /bin/bash -c \
  "for pci in 0000:0c:00.0 0000:2a:00.0 0000:41:00.0 0000:58:00.0 0000:86:00.0 0000:a5:00.0 0000:bd:00.0 0000:d5:00.0; do \
    echo \"=== PCI \$pci ===\"; \
    cat /host/sys/bus/pci/devices/\$pci/sriov_numvfs 2>/dev/null || echo 'File not found'; \
  done"
```
**Expected Result:** All 8 PCI devices should show `sriov_numvfs = 1`

### 2. Verify Daemon Logs Show VF Creation
```bash
kubectl logs sriov-network-config-daemon-cp8wn -n nvidia-network-operator --all-containers=true --tail=100 | \
  grep -E "(VF|vf|create|Create|configSriovDevice)"
```
**Expected Log Messages:**
- `configSriovDevice(): configure sriov device`
- `createVFs(): configure VFs for device`
- `VFIsReady()` for each VF
- `SetVfAdminMac()` for each VF

### 3. Verify Device Plugin Detected VFs
```bash
kubectl logs -n nvidia-network-operator \
  $(kubectl get pods -n nvidia-network-operator -l app=sriov-device-plugin \
    --field-selector spec.nodeName=10.0.65.77 -o name | head -1) --tail=30
```
**Expected Log Messages:**
- `selector index 0 will register 8 devices`
- `device added: [identifier: 0000:XX:00.1, ...]` for all 8 VFs
- `New resource server is created for sriov-rdma-vf ResourcePool`
- `Plugin: nvidia.com_sriov-rdma-vf.sock gets registered successfully at Kubelet`

### 4. Verify Node Capacity and Allocatable Resources
```bash
kubectl get node 10.0.65.77 -o jsonpath='{.status.capacity.nvidia\.com/sriov-rdma-vf}{"\n"}{.status.allocatable.nvidia\.com/sriov-rdma-vf}{"\n"}'
```
**Expected Result:** Both should show `8`

### 5. Verify Node Annotations
```bash
kubectl get sriovnetworknodestate 10.0.65.77 -n nvidia-network-operator -o jsonpath='{.metadata.annotations.sriovnetwork\.openshift\.io/desired-state}' && echo
```
**Expected Result:** Should show `Idle` (not `Reboot_Required`)

### 6. Verify SriovNetworkNodeState Status
```bash
kubectl get sriovnetworknodestate 10.0.65.77 -n nvidia-network-operator -o yaml | \
  grep -A 5 "syncStatus\|desired-state\|current-state"
```
**Expected Result:**
- `syncStatus: Succeeded` (or `InProgress` transitioning to `Succeeded`)
- `desired-state: Idle`
- `current-state: Idle`

## Final Status

After applying the fixes:

✅ **VFs Created:** All 8 PCI devices have `sriov_numvfs = 1`  
✅ **Node Capacity:** `nvidia.com/sriov-rdma-vf: 8`  
✅ **Node Allocatable:** `nvidia.com/sriov-rdma-vf: 8`  
✅ **Device Plugin:** Registered all 8 VFs with kubelet  
✅ **Daemon Status:** No longer requiring reboot, VFs created successfully  
✅ **Annotations:** Cleared, showing `Idle` state  

## Important Notes

1. **GRUB Configuration Priority:** On OCI HPC nodes, `/etc/default/grub.d/70-oci-hpc.cfg` takes precedence over `/etc/default/grub`. Always modify the OCI HPC config file for kernel arguments.

2. **Reboot Still Required:** While we cleared the annotations to allow VF creation, the node should still be rebooted to ensure the kernel boots with `ib_core.netns_mode=0` for persistence across reboots.

3. **Annotation Clearing:** The `Reboot_Required` annotations can block VF creation even when kernel arguments are correct. If VFs aren't being created after fixing kernel arguments, check and clear stale annotations.

4. **Device Plugin Restart:** After VFs are created, the device plugin may need to restart to detect them. This happened automatically in this case (device plugin pod restarted 27 seconds after VF creation).

## Prevention

To prevent this issue in the future:

1. **Ensure OCI HPC Config Includes Required Args:** The `/etc/default/grub.d/70-oci-hpc.cfg` file should include all required kernel arguments, including `ib_core.netns_mode=0` for RDMA/SRIOV configurations.

2. **Monitor Annotations:** Regularly check node and SriovNetworkNodeState annotations to ensure they don't get stuck in `Reboot_Required` state.

3. **Verify After Reboot:** After rebooting nodes, verify that:
   - Kernel command line includes required arguments (`cat /proc/cmdline`)
   - VFs are created (`cat /sys/bus/pci/devices/*/sriov_numvfs`)
   - Node shows correct capacity and allocatable resources

## Related Files

- `network-operator-values.yaml` - Helm values for network operator
- `bm.gpu.mi300x-sriov-policy.yaml` - SRIOV node policy defining VF requirements
- `vfconfig.yaml` - VF configuration daemonset
