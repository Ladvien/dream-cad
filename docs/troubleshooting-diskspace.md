# Disk Space Troubleshooting Guide

## Current Issue
System has only 3.1GB free space available, but MVDream requires minimum 50GB for:
- Model downloads (~10GB per model)
- Generated outputs
- Temporary files during generation
- Python dependencies

## Solutions

### Option 1: Free Up Space on Main Drive
```bash
# Clean package cache
sudo pacman -Scc

# Remove orphaned packages
sudo pacman -Rns $(pacman -Qtdq)

# Clean journal logs (keep last 2 weeks)
sudo journalctl --vacuum-time=2weeks

# Find large files
du -h / 2>/dev/null | sort -rh | head -30

# Clean user cache
rm -rf ~/.cache/pip
rm -rf ~/.cache/poetry
```

### Option 2: Use External Storage
```bash
# Mount external drive (example)
sudo mkdir /mnt/external
sudo mount /dev/sdX1 /mnt/external

# Create mvdream on external drive
mkdir -p /mnt/external/mvdream
ln -s /mnt/external/mvdream ~/mvdream-external

# Update paths in configuration
```

### Option 3: Expand Partition
If using LVM or have unallocated space:
```bash
# Check current partitions
lsblk
df -h

# Use GParted for GUI partition management
sudo pacman -S gparted
sudo gparted
```

### Option 4: Use Network Storage
Mount NAS or network drive:
```bash
# NFS mount example
sudo mkdir /mnt/nas
sudo mount -t nfs server:/path/to/share /mnt/nas

# SMB mount example
sudo mount -t cifs //server/share /mnt/nas -o username=user
```

## Temporary Workaround
For testing with limited space:
1. Use smaller models
2. Clean outputs after each generation
3. Use external USB drive for models

## Verification
After freeing space, verify with:
```bash
df -h ~/
python3 ~/mvdream/scripts/verify_requirements.py
```

## Prevention
- Set up automated cleanup scripts
- Monitor disk usage regularly
- Configure model cache limits
- Use symbolic links for large directories