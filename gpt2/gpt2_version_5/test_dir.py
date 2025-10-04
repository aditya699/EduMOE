import os, shutil, time
from pathlib import Path

# ====== SET THIS EXACTLY ONCE ======
OUTPUT_DIR = Path(r"D:\Desktop\2025_1\EduMOE\gpt2\gpt2_version_5\processed")
# ===================================

# 1) Resolve & create the folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
resolved = OUTPUT_DIR.resolve()
drive, _ = os.path.splitdrive(str(resolved))

print(f"✅ Using absolute output dir:\n  {resolved}")
print(f"📀 Drive selected: {drive or '(none?)'}")

# 2) Show free space on that drive
total, used, free = shutil.disk_usage(drive + "\\")
gb = 1024**3
print(f"💾 Free space on {drive}: {free/gb:.2f} GB (total: {total/gb:.2f} GB)")

# 3) Canary write (1 MiB) to guarantee we’re writing to the right place
canary = resolved / "__write_test__.tmp"
blob = b"\0" * (1024 * 1024)  # 1 MiB
with open(canary, "wb") as f:
    f.write(blob)
    f.flush()
    os.fsync(f.fileno())

print(f"🧪 Wrote canary: {canary}  (size={canary.stat().st_size/1024/1024:.1f} MiB)")
time.sleep(0.25)

# 4) Clean up
canary.unlink(missing_ok=True)
print("🧹 Removed canary file.")
print("✅ Write test succeeded. This is where train.bin / val.bin will be created.")
