# LLM-D Deployments on MI300X AMD GPUs

## Pre-reqs because of funky node stuff

This dir covers setup and deployment of llm-d on AMD clusters on OCI OKE.

The most important part of this setup to do first is to correct any OCI networking issues which are incompatible with NIXL.

From the llm-d team:

> @Vincent Cave
>
> OCI networking TC mapping: TC --> prio
> 0 --> prio1
> 41 --> prio0
> 106 --> prio5

> so ROCSHMEM seems to look at ROCSHMEM_GDA_TRAFFIC_CLASS=41 then figure that the DSCP value of 10 is mapped to prio0 and so send traffic on the prio0 queue. But it does not seem to set the TC in the packet headers.

> when this happens, on the prefill nodes, the TX bytes show up on prio0 queue - since traffic is sent out from that queue. But the RX bytes show up on prio1 queue, because network sees TC=0 in the packets and maps it to prio1.

> You can see this in this before fix log (edited) 
> [4:54 PM]The fix: 

> force all RDMA traffic to use TC=41. I did the following for all backend NICs across the cluster:
> echo 41 > /sys/class/infiniband/mlx5_0/tc/1/traffic_class

> Now, we got all the traffic (DeepEP high throughput, RIXL, Mori low latency all2all) to land on prio0.

> You can see this in this after fix log. (edited)

ROCSHMEM version which fixes this: TBD


Additionally, some ip routing rules causes by older versions of the OCA were identified in joint troubleshooting with OCI and AMD. Using a newer version of OCA 1.54 will resolve long term, but if your node returns values from the following:

```bash
ip rule | grep "^10:" | grep "from all to 10.224" | awk -F: '{print $2}'
from all to 10.224.14.230 lookup 10
	from all to 10.224.30.230 lookup 11
	from all to 10.224.46.230 lookup 12
	from all to 10.224.62.230 lookup 13
	from all to 10.224.78.230 lookup 14
	from all to 10.224.94.230 lookup 15
	from all to 10.224.110.230 lookup 16
	from all to 10.224.126.230 lookup 17
```
The rules need to be deleted:
```bash
ip rule | grep "^10:" | grep "from all to 10.224" | awk -F: '{print $2}' | while IFS= read -r line ; do sudo ip rule delete $line ; done
```
Canonical-Ubuntu-22.04-2025.10.31-0-OCA-DOCA-OFED-3.1.0-AMD-ROCM-710-2025.12.08

https://github.com/oci-ai-incubations/oci-hpc-images-copy/pull/4/changes#r2808095392


## Files and helms needed for llm-d setup

First, checkout the benchmarking repo:
```bash
git clone https://github.com/tlrmchlsmth/j-llm-d.git
cd j-llm-d/
git checkout glm-4.7
cd glm-pareto/
git submodule update --init --recursive
```

Next, install `just` for running:
```bash
mkdir -p ~/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
```

Next, `helmfile`:
```bash
wget https://github.com/helmfile/helmfile/releases/download/v1.2.3/helmfile_1.2.3_linux_amd64.tar.gz
tar -xzf helmfile_1.2.3_linux_amd64.tar.gz
mv helmfile ~/bin

# Helm 4
helm plugin install https://github.com/databus23/helm-diff --verify=false

# Helm 3
helm plugin install https://github.com/databus23/helm-diff
```

Next, install tools for llm-d:
```bash
cd j-llm-d/llm-d/guides/prereq/gateway-provider/
./install-gateway-provider-dependencies.sh
helmfile apply -f istio.helmfile.yaml
kubectl api-resources --api-group=inference.networking.k8s.io

# ServiceMonitoring and PodMonitoring
cd ~
git clone https://github.com/prometheus-operator/prometheus-operator.git
cd prometheus-operator/
k create -f bundle.yaml
```

Finally, run the bench:
```bash
cd ~/j-llm-d/glm-pareto
just run disagg-1p1d-8
```


## Notes, patches, bugs
Deep-EP needs:

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
        -DGPU_TARGETS=$GFX_COMPILATION_ARCHEnvironment variables used:
    export ROCSHMEM_HEAP_SIZE=8589934592
    export ROCSHMEM_MAX_NUM_CONTEXTS=64
    export ROCSHMEM_BACKEND=gda
    export ROCSHMEM_GDA_TRAFFIC_CLASS=41
    export ROCSHMEM_HCA_LIST=^mlx5_1,mlx5_6
```