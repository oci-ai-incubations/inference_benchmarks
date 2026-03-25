# Simple guide to raiding NVMe disks when you start a new system

Why RAID? Redundant Array of Independent Disks (RAID) takes a group of independent disks, and creates a shared storage pool from those disks. The general purpose of this is 2 fold:
1. Performance
2. Redundancy

Take super fast NVMe, as an example. RAID'ing multiple disks at RAID0 will give you (roughly) the aggregate throughput performance of all the disks combined. However, in the case of RAID0, if a single disk fails, all the data is corrupt since pieces of that data are striped across all RAID disks in the configuration. RAID1 halves the effective capacity, but gives redundancy because each chunk of data is striped to 2 disks. This way, if a disk fails, the hope is that data can be recovered from its pair.

Below, I'll setup a raid0 array because it offers the best performance. A note about raid0 is that it offers the highest performance, but it offers no redundancy. If data is lost on any drive, data is lost in the whole array. It is the best for short, ephemeral data. If datta needs to persist, consider mounting a colder (slower) storage for long term storage. Object storage via NFS is a great use case for this.

1. Install mdadm if not already installed
```bash
sudo apt-get update
sudo apt-get install mdadm
```
2. Create RAID 0 array using default chunk size. If working with very large or very small data, tuning this may offer better performance:
```bash
lsblk 
NAME    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0     7:0    0  55.7M  1 loop /snap/core18/2823
loop1     7:1    0  55.7M  1 loop /snap/core18/2829
loop2     7:2    0  63.9M  1 loop /snap/core20/2318
loop3     7:3    0    64M  1 loop /snap/core20/2379
loop4     7:4    0    87M  1 loop /snap/lxd/28373
loop5     7:5    0    87M  1 loop /snap/lxd/29351
loop6     7:6    0  77.3M  1 loop /snap/oracle-cloud-agent/72
loop7     7:7    0  78.6M  1 loop /snap/oracle-cloud-agent/x1
loop8     7:8    0  38.8M  1 loop /snap/snapd/21759
sda       8:0    0   256G  0 disk 
├─sda1    8:1    0 255.9G  0 part /
├─sda14   8:14   0     4M  0 part 
└─sda15   8:15   0   106M  0 part /boot/efi
nvme0n1 259:0    0   3.5T  0 disk 
nvme3n1 259:1    0   3.5T  0 disk 
nvme2n1 259:2    0   3.5T  0 disk 
nvme1n1 259:3    0   3.5T  0 disk 
nvme4n1 259:4    0   3.5T  0 disk 
nvme5n1 259:5    0   3.5T  0 disk 
nvme6n1 259:6    0   3.5T  0 disk 
nvme7n1 259:7    0   3.5T  0 disk 
```
You need to use the nvme names prefixed with `/dev/` for the raid. 
```bash
sudo mdadm --create --verbose /dev/md0 --level=0 --raid-devices=8 /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1
### stdout
mdadm: chunk size defaults to 512K
mdadm: Defaulting to version 1.2 metadata
mdadm: array /dev/md0 started.
```
3. Create a filesystem on the RAID array (ext4 here)
```bash
sudo mkfs.ext4 /dev/md0
### stdout
mke2fs 1.46.5 (30-Dec-2021)
Discarding device blocks: done                            
Creating filesystem with 7501211648 4k blocks and 468826112 inodes
Filesystem UUID: 185e5b79-d63b-4dac-a600-1e6944937983
Superblock backups stored on blocks: 
	32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208, 
	4096000, 7962624, 11239424, 20480000, 23887872, 71663616, 78675968, 
	102400000, 214990848, 512000000, 550731776, 644972544, 1934917632, 
	2560000000, 3855122432, 5804752896

Allocating group tables: done                            
Writing inode tables: done                            
Creating journal (262144 blocks): done
Writing superblocks and filesystem accounting information: done         
```
4. Create a mount - I like to be verbose with the top level mount.
```bash
sudo mkdir /mnt/nvme
```
5. Mount the new fs
```bash
sudo mount /dev/md0 /mnt/nvme
```
6. Check the storage:
```bash
df -h /mnt/nvme/
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         28T   24K   27T   1% /mnt/nvme
```
6. Make the mount persistent across reboots by adding to /etc/fstab
```bash
echo '/dev/md0 /mnt/nvme ext4 defaults 0 0' | sudo tee -a /etc/fstab
```
7. Save the raid config
```bash
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
```
8. Update the initramfs
```bash
sudo update-initramfs -u
```

