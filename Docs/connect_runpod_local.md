🚀 RunPod + VS Code + Jupyter Setup Guide

This guide shows how to connect your RunPod Jupyter Notebook Pod to VS Code on your local machine so you can code locally while running on RunPod GPU.

1️⃣ Generate SSH Keys (on your local machine)

Open PowerShell and run:

ssh-keygen -t ed25519 -C "meenakshi.bhtt@gmail.com"


When it asks for location → press Enter (default: C:\Users\<YourName>\.ssh\id_ed25519).

When it asks for a passphrase → press Enter twice (skip).

✅ This creates two files:

id_ed25519 → private key (do not share)

id_ed25519.pub → public key (safe to share)

2️⃣ Copy Your Public Key

Show the public key in PowerShell:

cat $env:USERPROFILE\.ssh\id_ed25519.pub


Output will look like:

ssh-ed25519 AAAAC3Nz...long_string... meenakshi.bhtt@gmail.com

3️⃣ Add Public Key to RunPod

Go to RunPod Dashboard → SSH Settings.

Paste the entire public key string.

Save.

This allows RunPod to accept SSH connections from your machine.

4️⃣ Connect to RunPod via SSH (optional test)

You can now SSH into your pod:

ssh -p <PORT> root@<POD_ID>.proxy.runpod.net


Replace <PORT> with the one shown under Direct TCP Ports (e.g., 19123).

Replace <POD_ID> with your pod ID (e.g., tt956fh5rktgga).

If setup is correct → you’ll log in without any password.

5️⃣ Get Your Jupyter Token

In RunPod Dashboard → Connect → Jupyter Lab.

It opens a URL in browser like:

https://tt956fh5rktgga-8888.proxy.runpod.net/lab?token=8c9d7f2a6e54321abc...


Copy the full URL (including ?token=...).

The string after token= is your Jupyter token.

6️⃣ Connect VS Code to RunPod Jupyter

In VS Code, open Command Palette (Ctrl+Shift+P).

Search for → Jupyter: Specify Jupyter Server for Connections.

Choose → Existing: Specify the URL of an existing server.

Paste your full Jupyter URL:

https://tt956fh5rktgga-8888.proxy.runpod.net/lab?token=8c9d7f2a6e54321abc...


Confirm. Now VS Code connects directly to your RunPod Jupyter kernel.

7️⃣ Handling Password Prompt

If VS Code asks for a password, paste the token value (just the random string after token=).

If it connects via URL, you won’t see a password prompt at all.

✅ Summary

SSH keys allow secure login (no passwords on RunPod).

Jupyter uses a token for authentication, not a password.

In VS Code:

Use SSH key for remote file/terminal access.

Use Jupyter token for connecting notebooks.

Now you have a smooth setup:

Edit code locally in VS Code.

Run everything on RunPod GPU.

No more password issues.