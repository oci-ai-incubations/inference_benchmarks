# The Hidden Infrastructure Problem in Distributed LLM Inference

You've heard the pitch: scale your LLM across multiple GPUs, across multiple nodes, and watch throughput climb. What nobody tells you is that the path between your GPU and the network is a minefield of hardware topology decisions — and Kubernetes, the platform you're running all of this on, is almost entirely blind to them.

This post traces the physical journey of data through a GPU node, explains why that journey matters for distributed inference, and walks through the layers of Kubernetes abstraction that can silently destroy your performance. We'll end with what the community is building to fix it.

---

## Part 1: Inside the Node — How a GPU Talks to the World

### The CPU–PCIe–GPU Connection

Before any data reaches your model, it has to travel a physical path. On a modern server, a GPU doesn't sit directly on the CPU. It's connected through the **PCIe (Peripheral Component Interconnect Express)** bus — a high-speed serial interconnect that serves as the highway between the CPU, memory, and every accelerator or I/O device in the system.

Here's what that looks like on a single-socket machine with one GPU:

```
┌──────────────────────────────────────────────┐
│                    CPU                        │
│              ┌────────────┐                   │
│              │   Memory   │                   │
│              │  Controller│                   │
│              └─────┬──────┘                   │
│                    │                          │
│           ┌────────┴────────┐                 │
│           │  PCIe Root      │                 │
│           │  Complex        │                 │
│           └───┬─────────┬───┘                 │
│               │         │                     │
│          ┌────┴───┐ ┌───┴────┐                │
│          │  GPU   │ │  NIC   │                │
│          └────────┘ └────────┘                │
└──────────────────────────────────────────────┘
```

When your application sends a tensor to the GPU, it moves from system memory → through the PCIe root complex → across a PCIe link → into GPU memory (VRAM). The reverse happens when the GPU writes results back. PCIe Gen5 x16 delivers around 64 GB/s of bandwidth in each direction, and every hop through the root complex adds a small amount of latency.

This path is fine for a single GPU. The problems start when you have *eight* of them.

### NUMA: When One CPU Isn't Enough

High-end GPU servers don't have one CPU — they have two (or more). Each CPU has its own memory controller, its own PCIe root complex, and its own set of directly attached devices. This architecture is called **NUMA (Non-Uniform Memory Access)**.

The "non-uniform" part is the key: a CPU can access its *own* memory and *its own* PCIe devices quickly, but accessing the *other* CPU's memory or devices requires crossing an inter-socket link (UPI on Intel, xGMI/Infinity Fabric on AMD). That crossing adds latency and reduces bandwidth.

On an AMD MI300x bare-metal node with 8 GPUs, the layout is symmetric across two sockets:

```
┌─────────────── NUMA Node 0 ──────────────────┐  ┌─────────────── NUMA Node 1 ──────────────────┐
│                                               │  │                                               │
│                 CPU Socket 0                  │  │                 CPU Socket 1                  │
│           ┌──────────────────┐                │  │           ┌──────────────────┐                │
│           │  Memory + PCIe   │                │  │           │  Memory + PCIe   │                │
│           │  Root Complex    │                │  │           │  Root Complex    │                │
│           └──┬──┬──┬──┬──┬──┘                │  │           └──┬──┬──┬──┬──┬──┘                │
│              │  │  │  │  │                    │  │              │  │  │  │  │                    │
│    ┌─────┐ ┌┴┐┌┴┐┌┴┐┌┴┐┌┴─────┐              │  │    ┌─────┐ ┌┴┐┌┴┐┌┴┐┌┴┐┌┴─────┐              │
│    │GPU 0│ │1││2││3││ ││NIC×4 │              │  │    │GPU 4│ │5││6││7││ ││NIC×4 │              │
│    └─────┘ └─┘└─┘└─┘└─┘└──────┘              │  │    └─────┘ └─┘└─┘└─┘└─┘└──────┘              │
│                                               │  │                                               │
└───────────────────────┬───────────────────────┘  └───────────────────────┬───────────────────────┘
                        │          UPI / xGMI Inter-Socket Link           │
                        └─────────────────────────────────────────────────┘
```

GPUs 0–3 and NICs 0–3 live on Socket 0 (NUMA node 0). GPUs 4–7 and NICs 4–7 live on Socket 1 (NUMA node 1). If GPU 0 needs to talk to a NIC on Socket 1, the data has to cross the inter-socket link — adding latency and cutting bandwidth. For RDMA workloads pushing hundreds of gigabits, this penalty is severe.

### PCIe Switches: The Topology Below NUMA

NUMA is not the finest granularity. Within each socket, GPUs and NICs aren't all wired directly to the CPU's root complex. They're grouped behind **PCIe switches** — small silicon chips that aggregate multiple devices onto a shared upstream link to the CPU.

On the MI300x, the layout under each socket looks roughly like this:

```
           CPU Socket 0 Root Complex
           ┌───┬───┬───┬───┐
           │   │   │   │   │
         ┌─┴─┐┌┴──┐┌┴──┐┌─┴─┐
         │SW0││SW1││SW2││SW3│    (PCIe Switches)
         └┬─┬┘└┬─┬┘└┬─┬┘└┬─┬┘
          │ │  │ │  │ │  │ │
        GPU0│GPU1│GPU2│GPU3│
           NIC0 NIC1 NIC2 NIC3
```

GPU 0 and NIC 0 sit behind the same PCIe switch (SW0). Communication between them is **PIX** — peer-to-peer through the switch, the fastest possible path. GPU 0 talking to NIC 1 has to go up through SW0, through the root complex, and back down through SW1 — that's **PXB**, slower but still on the same socket. GPU 0 talking to NIC 4 on the other socket? That's a full inter-socket crossing — the worst case.

This hierarchy matters because, as we'll see, Kubernetes currently understands NUMA but has no concept of PCIe switches.

---

## Part 2: Crossing the Wire — GPU-to-GPU with RDMA

### Why RDMA Exists

Distributed LLM inference splits a model across multiple GPUs, often across multiple physical nodes. Techniques like tensor parallelism, pipeline parallelism, and disaggregated KV-cache transfer all require GPUs to exchange large amounts of data at high frequency.

The traditional network path looks like this:

```
GPU → PCIe → CPU → Memory → TCP/IP Stack → NIC → Wire → NIC → TCP/IP Stack → Memory → CPU → PCIe → GPU
```

Every step through the CPU and TCP/IP stack adds latency and consumes CPU cycles. For the message sizes and frequencies involved in distributed inference (hundreds of megabytes, hundreds of times per second), this path is unacceptable.

**RDMA (Remote Direct Memory Access)** eliminates the CPU from the data path entirely. With RDMA-capable NICs (often Mellanox/NVIDIA ConnectX cards), the data path becomes:

```
┌──────────┐                                           ┌──────────┐
│  GPU 0   │                                           │  GPU 4   │
│ (Node A) │                                           │ (Node B) │
└────┬─────┘                                           └────┬─────┘
     │ PCIe                                                 │ PCIe
     │                                                      │
┌────┴─────┐          RDMA (kernel bypass)            ┌────┴─────┐
│  RDMA    │ ════════════════════════════════════════> │  RDMA    │
│  NIC     │         Wire (100-400 Gbps)              │  NIC     │
└──────────┘                                          └──────────┘
```

With technologies like **GPUDirect RDMA**, the NIC can read directly from GPU memory over PCIe without any CPU involvement. The data goes: GPU VRAM → PCIe → NIC → wire → NIC → PCIe → GPU VRAM. No copies, no kernel transitions, no TCP overhead.

This is why every serious GPU cluster uses RDMA for inter-node communication, and why libraries like NCCL (NVIDIA) and RCCL (AMD) exist — they orchestrate these transfers across collective operations like all-reduce and all-gather.

### The Pairing Problem

But notice what RDMA requires: a **tight physical relationship between the GPU and the NIC**. If GPU 0 is on NUMA 0 and the NIC is on NUMA 1, the "direct" path still has to cross the inter-socket link. You've eliminated the CPU from the *software* path, but you've put a socket hop in the *hardware* path.

The optimal pairing is GPU and NIC on the same PCIe switch. The acceptable pairing is same NUMA node. The disastrous pairing is cross-NUMA. And on a node with 8 GPUs and 8 NICs, there are many ways to get the pairing wrong.

---

## Part 3: Enter Kubernetes — Where Topology Goes to Die

### GPUs as Resources

Kubernetes doesn't know what a GPU is. To the scheduler, your 8 GPUs are just a number:

```yaml
allocatable:
  amd.com/gpu: "8"
```

A **device plugin** (AMD's `amd-gpu-device-plugin` or NVIDIA's equivalent) runs on each node, discovers the GPUs, and reports them to the kubelet. When a pod requests `amd.com/gpu: 4`, the device plugin picks 4 GPUs and passes their device paths to the container.

Which 4? That depends on the device plugin's internal allocation logic. The Kubernetes scheduler has no opinion — it only checks that the number is available.

### The hostNetwork Temptation

Before SR-IOV and Multus entered the picture, the common approach to giving a GPU pod access to the backend RDMA network was simple: `hostNetwork: true`.

```yaml
spec:
  hostNetwork: true
  containers:
  - name: training
    resources:
      limits:
        amd.com/gpu: 8
```

This gives the pod the host's entire network stack — every interface, every IP, every route. It works. The RDMA NICs are visible, NCCL can find them, and collective operations run at full speed.

But it comes with serious problems:

```
┌──────────── Host Network Namespace ─────────────┐
│                                                  │
│   Pod A (hostNetwork: true)                      │
│   ┌────────────────────────────────────────────┐ │
│   │  Sees ALL interfaces:                      │ │
│   │  - eth0 (management)                       │ │
│   │  - rdma0, rdma1, rdma2, rdma3              │ │
│   │  - rdma4, rdma5, rdma6, rdma7              │ │
│   │  - ... every other interface               │ │
│   │                                            │ │
│   │  Shares ports with host and other pods     │ │
│   │  Has access to host management network     │ │
│   └────────────────────────────────────────────┘ │
│                                                  │
│   Pod B (hostNetwork: true)                      │
│   ┌────────────────────────────────────────────┐ │
│   │  Sees the SAME interfaces                  │ │
│   │  Port conflicts with Pod A                 │ │
│   │  No isolation whatsoever                   │ │
│   └────────────────────────────────────────────┘ │
│                                                  │
│   kubelet, kube-proxy, system services...        │
│   (all sharing the same network namespace)       │
│                                                  │
└──────────────────────────────────────────────────┘
```

**No isolation.** Every pod sees every interface, including the management network. A misconfigured application can interfere with cluster control plane traffic.

**Port conflicts.** Two pods can't both bind to the same port. You're back to the bare-metal problem of port management that Kubernetes was supposed to solve.

**No multi-tenancy.** You can't run two independent workloads on the same node, each with their own set of RDMA NICs. Everyone shares everything.

**Security exposure.** The pod has direct access to the host's network stack. Any network-level vulnerability in the application is a host-level vulnerability.

**No resource accounting.** Kubernetes has no idea which NICs the pod is using. There's no way to limit, track, or schedule network resources.

For a single-purpose training cluster where one job owns the entire node, `hostNetwork` works. For anything resembling a shared, multi-tenant production environment, it's a non-starter.

---

## Part 4: SR-IOV — NICs as First-Class Resources

### What SR-IOV Is

**SR-IOV (Single Root I/O Virtualization)** is a PCIe specification that allows a single physical NIC (a "Physical Function" or PF) to present itself as multiple virtual NICs ("Virtual Functions" or VFs). Each VF is a real PCIe device with its own BDF address, its own interrupts, and its own DMA engine. The hardware does the multiplexing — no software switch in the path.

```
┌────────────────────────────────────┐
│     Physical NIC (PF)              │
│     e.g., Mellanox ConnectX-7      │
│                                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │ VF0 │ │ VF1 │ │ VF2 │ │ VF3 │  │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
│     │       │       │       │      │
└─────┼───────┼───────┼───────┼──────┘
      │       │       │       │
    Pod A   Pod B   Pod C   Pod D
```

Each VF gets passed directly into a pod's network namespace via a CNI plugin like **Multus** (which allows pods to have multiple network interfaces) and the **SR-IOV CNI**. The pod gets a dedicated NIC with near-hardware-line-rate performance, complete isolation from other pods, and — critically for RDMA — direct access to the NIC's RDMA capabilities.

### NICs as Kubernetes Resources

With the **SR-IOV Network Operator** running in the cluster, VFs are created on each node and reported to the kubelet through a device plugin, just like GPUs:

```yaml
allocatable:
  amd.com/gpu: "8"
  nvidia.com/sriov-rdma-vf: "8"
```

Now a pod can request specific numbers of each:

```yaml
resources:
  limits:
    amd.com/gpu: 4
    nvidia.com/sriov-rdma-vf: 4
```

This is a massive improvement over `hostNetwork`. Each pod gets exactly the NICs it needs, with full isolation, proper resource accounting, and no port conflicts. Kubernetes can schedule based on NIC availability.

But there's a catch.

### Two Device Plugins, Zero Coordination

The GPU device plugin allocates GPUs. The SR-IOV device plugin allocates VFs. They run independently, report to the kubelet independently, and allocate independently.

When a pod requests 4 GPUs and 4 VFs, here's what happens:

```
        kubelet
       ┌───┴───┐
       │       │
   ┌───┴───┐ ┌┴────────┐
   │ AMD   │ │ SR-IOV   │
   │ GPU   │ │ Device   │
   │Plugin │ │ Plugin   │
   └───┬───┘ └────┬─────┘
       │          │
  "Here are 4    "Here are 4
   GPUs from      VFs from
   NUMA 0"        NUMA 1"        <-- No coordination!
```

The GPU plugin might hand out GPUs 0–3 (NUMA 0). The SR-IOV plugin, knowing nothing about the GPU allocation, might hand out VFs 4–7 (NUMA 1). Every RDMA transfer now crosses the inter-socket link. You've done everything right — SR-IOV, RDMA, dedicated VFs — and you're still getting terrible performance because of a topology mismatch that neither plugin knows about.

---

## Part 5: Topology Manager — Teaching Kubernetes About NUMA

### How It Works

Kubernetes *does* have a mechanism for coordinating device allocation across plugins: the **kubelet Topology Manager**. It's not enabled by default, and it's not widely known, but it solves the NUMA alignment problem.

When Topology Manager is enabled with the `restricted` policy, the kubelet's allocation flow changes:

```
        Pod requests: 4 GPUs + 4 VFs
                    │
                    ▼
        ┌───────────────────────┐
        │   Topology Manager    │
        │                       │
        │  "AMD GPU plugin,     │
        │   what NUMA nodes     │
        │   can you satisfy     │
        │   4 GPUs from?"       │
        │                       │
        │   → NUMA 0 or NUMA 1  │
        │                       │
        │  "SR-IOV plugin,      │
        │   what NUMA nodes     │
        │   can you satisfy     │
        │   4 VFs from?"        │
        │                       │
        │   → NUMA 0 or NUMA 1  │
        │                       │
        │  Intersection:         │
        │   NUMA 0 ∩ NUMA 0 = ✓ │
        │   NUMA 1 ∩ NUMA 1 = ✓ │
        │                       │
        │  Pick NUMA 0.          │
        │  Both plugins MUST     │
        │  allocate from NUMA 0. │
        └───────────────────────┘
```

Both plugins are now forced to allocate from the same NUMA node. GPUs 0–3 pair with VFs 0–3 — all on NUMA 0, all behind the same socket's PCIe root complex.

### Enabling It

The kubelet configuration (`/etc/kubernetes/kubelet-config.json`) needs four additions:

```json
{
  "topologyManagerPolicy": "restricted",
  "topologyManagerScope": "container",
  "topologyManagerPolicyOptions": {
    "prefer-closest-numa-nodes": "true"
  },
  "cpuManagerPolicy": "static"
}
```

**`restricted`** means: if the resources can fit on a single NUMA node, force it. If they genuinely can't (e.g., 8 GPUs + 8 VFs span both sockets), allow it. This is the right balance — `single-numa-node` would reject legitimate full-node requests.

**`prefer-closest-numa-nodes`** is critical and defaults to `false`. Without it, the Topology Manager doesn't actually prefer NUMA-local allocations when multiple options are equally valid.

**`cpuManagerPolicy: static`** is required for Topology Manager to function. It enables CPU pinning for guaranteed QoS pods, which is a prerequisite for the topology-aware allocation machinery.

### What It Solves (and What It Doesn't)

Topology Manager eliminates cross-socket GPU-NIC pairings. On the MI300x nodes we tested, enabling it with the `restricted` policy and `prefer-closest-numa-nodes` resulted in perfect NUMA alignment across all 8 pods in a 1-GPU-1-NIC test deployment: 4 pods landed on NUMA 0, 4 on NUMA 1, with each pod's GPU and VF on the same socket.

But NUMA is the coarsest level of topology. Within NUMA node 0, there are 4 PCIe switches. GPU 0 and NIC 0 share SW0 (PIX — optimal). GPU 0 and NIC 2 are both on NUMA 0, but behind different switches — the data has to traverse up to the root complex and back down (PXB — acceptable, but not optimal).

Topology Manager has no concept of PCIe switches. It got us from "random, potentially cross-socket" to "guaranteed same-NUMA." That's the big win — the cross-socket penalty can cut RDMA bandwidth by 30–50%. The within-NUMA-different-switch penalty is single-digit microseconds of additional latency. For LLM inference workloads, the socket alignment is what matters.

But the community isn't stopping here.

---

## Part 6: The Future — DRA and DRANET

### Dynamic Resource Allocation (DRA)

Kubernetes v1.26 introduced **Dynamic Resource Allocation (DRA)** as an alpha feature, and it's been progressing toward GA (targeted for v1.35). DRA replaces the device plugin model with something far more expressive.

Instead of device plugins reporting simple counts (`amd.com/gpu: 8`) to the kubelet, DRA drivers publish **ResourceSlices** — rich descriptions of available devices with arbitrary attributes. A GPU isn't just a number anymore; it has a PCI address, a NUMA node, a PCIe root complex identifier, available memory, and any other attribute the driver wants to expose.

Pods request resources through **ResourceClaims** and **ResourceClaimTemplates** with CEL (Common Expression Language) selectors:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: gpu-with-aligned-nic
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly:
          deviceClassName: gpu.vendor.com
          count: 2
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

That last line — `matchAttribute: "resource.kubernetes.io/pcieRoot"` — is the breakthrough. It tells the scheduler: **the GPUs and NICs must share the same PCIe root complex.** Not just the same NUMA node — the same PCIe switch. This is the PIX-level alignment that Topology Manager can't express.

### DRANET: Network Interfaces as DRA Resources

**DRANET** (Dynamic Resource Allocation for Networking) is a Kubernetes-SIGs project that extends DRA to network interfaces. It treats NICs as first-class DRA devices with attributes like RDMA capability, PCI address, PCIe root, link speed, and more.

Where the SR-IOV device plugin says "there are 8 VFs available," DRANET says "here is VF `0000:0c:00.1` with RDMA capability, on PCIe root `pci0000:00`, with 400 Gbps link speed, on NUMA node 0." The scheduler can make intelligent placement decisions based on the full hardware topology.

Combined with NVIDIA's DRA GPU driver (or a future AMD equivalent), a pod spec can express what we've been building toward in this entire blog post: **give me N GPUs and N RDMA NICs, and make sure each GPU-NIC pair shares the same PCIe switch.**

```
┌────────────────────────────────────────────────┐
│           DRA-Aware Scheduling                  │
│                                                 │
│  ResourceClaim: 2 GPUs + 2 RDMA NICs           │
│  Constraint: matchAttribute pcieRoot            │
│                                                 │
│  Scheduler evaluates:                           │
│  ┌─────────────────────────────────────┐        │
│  │ Node ResourceSlices:                │        │
│  │                                     │        │
│  │  GPU 0 (pcieRoot: pci0000:00) ──┐   │        │
│  │  NIC 0 (pcieRoot: pci0000:00) ──┤   │ Match! │
│  │                                  │   │        │
│  │  GPU 1 (pcieRoot: pci0000:08) ──┐   │        │
│  │  NIC 1 (pcieRoot: pci0000:08) ──┤   │ Match! │
│  │                                  │   │        │
│  └─────────────────────────────────────┘        │
│                                                 │
│  Result: GPU 0 + NIC 0 (same switch)            │
│          GPU 1 + NIC 1 (same switch)            │
│                                                 │
└────────────────────────────────────────────────┘
```

Google announced managed DRANET in GKE in October 2025, launching with their A4X Max instances (NVIDIA GB300 NVL72). In their internal testing, NUMA-aligned NIC allocation improved bus bandwidth by up to 59.6%. The project is open source and moving toward broader adoption.

KEP-4381 proposes the standard `resource.kubernetes.io/pcieRoot` attribute, giving all DRA drivers a common vocabulary for expressing PCIe topology. When both the GPU DRA driver and DRANET publish this attribute, the scheduler can enforce PCIe-switch-level pairing — the finest granularity of hardware topology that matters for RDMA performance.

---

## The Journey So Far

Let's step back and look at the progression:

| Approach | GPU-NIC Topology | Isolation | Multi-Tenancy | Granularity |
|----------|-----------------|-----------|---------------|-------------|
| **hostNetwork** | Whatever NCCL finds | None | No | N/A |
| **SR-IOV + device plugins** | Random | Per-VF | Yes | None |
| **SR-IOV + Topology Manager** | Same NUMA | Per-VF | Yes | NUMA node |
| **DRA + DRANET** | Same PCIe switch | Per-VF | Yes | PCIe root |

Each step eliminates a class of performance-destroying misalignment:

- **hostNetwork → SR-IOV** gave us isolation and resource accounting.
- **SR-IOV → Topology Manager** eliminated cross-socket GPU-NIC pairings.
- **Topology Manager → DRA/DRANET** will eliminate cross-switch pairings within a socket.

The infrastructure beneath distributed LLM inference is not just "more GPUs." It's the silent topology decisions between those GPUs and the network that determine whether your cluster runs at 40% or 100% of its theoretical throughput. Kubernetes is learning to see this topology — and the gap between "it works" and "it works *well*" is finally closing.
