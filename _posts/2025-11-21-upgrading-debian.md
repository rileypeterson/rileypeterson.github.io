---
layout: post
title:  "Upgrading Debian"
date:   2025-11-21
categories: devops, linux
---

It's been a while since I've posted anything here, but I figured it was a good time to document this procedure since I may need to do it again.

I have a [linux machine]({% link _projects/linux/linux-dual-boot-debian.md %}) which needs upgrading.

```bash
# Check the current version
cat /etc/debian_version
```
```
11.11
```


This is Bullseye, which as of today (2025-11-21), is in LTS. We're going to upgrade to Trixie (13.2). Maybe the next time I do this they'll run out of Toy Story characters, but really I should be upgrading every major release. 

There is nothing super important to back up on this machine so I'm not going through the hassle of backing it up. I did disconnect a drive mounted to it:
```bash
# Identify the drive
df -h
sudo umount <Filesystem Name Identifier In Previous Command>
sudo sync
# Disconnect the USB
```

Begin the update. First we need to go from bullseye to bookworm:


```bash
sudo apt update
sudo apt full-upgrade
# List external repositories (may need to re-install these)
sudo apt list '?narrow(?installed, ?not(?origin(Debian)))'
```
```
Listing... Done
containerd.io/bullseye,now 2.1.5-1~debian.11~bullseye amd64 [installed]
docker-buildx-plugin/bullseye,now 0.30.0-1~debian.11~bullseye amd64 [installed]
docker-ce-cli/bullseye,now 5:29.0.2-1~debian.11~bullseye amd64 [installed]
docker-ce-rootless-extras/bullseye,now 5:29.0.2-1~debian.11~bullseye amd64 [installed]
docker-ce/bullseye,now 5:29.0.2-1~debian.11~bullseye amd64 [installed]
docker-compose-plugin/bullseye,now 2.40.3-1~debian.11~bullseye amd64 [installed]
docker-scan-plugin/bullseye,now 0.23.0~debian-bullseye amd64 [installed]
```
```bash
# Backup original sources lists
mkdir ~/apt
cp /etc/apt/sources.list ~/apt
cp -r /etc/apt/sources.list.d/ ~/apt

sudo sed -i 's/bullseye/bookworm/g' /etc/apt/sources.list
sudo nano /etc/apt/sources.list
# Remove The comments at the top which mention bullseye
sudo sed -i 's/bullseye/bookworm/g' /etc/apt/sources.list.d/*
# One of my lists in here goes back to stretch (debian v9) 
# so I will remove that at the end

sudo apt update
sudo apt upgrade --without-new-pkgs # (do ":q" to get out of changelogs)
sudo apt full-upgrade # (select Yes when asked about restarting services)
# There's diffs in sshd_config, I will just remake the changes which are in the
# link at the top of the page so I install maintainer's version.
sudo reboot

# SSH back in
cat /etc/debian_version
```
```
12.12
```
```bash
# Now we go from bookworm to trixie

cp /etc/apt/sources.list ~/apt
cp -r /etc/apt/sources.list.d/ ~/apt

sudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list
sudo sed -i 's/bookworm/trixie/g' /etc/apt/sources.list.d/*

sudo apt update
sudo apt upgrade --without-new-pkgs # (do ":q" to get out of changelogs)
sudo apt full-upgrade
sudo apt autoremove
sudo reboot

# SSH back in 
cat /etc/debian_version
```
```
13.2
```
```bash
# Removing old package in /etc/apt/sources.list.d/<PACKAGE>.list
sudo rm /etc/apt/sources.list.d/<PACKAGE>.list
sudo rm /etc/apt/trusted.gpg.d/<PACKAGE>.asc
sudo apt update
sudo apt autoremove

# Now go back and reinstate "PermitRootLogin no" in /etc/ssh/sshd_config
sudo nano /etc/ssh/sshd_config
# Also, I set X11Forwarding to no
sudo systemctl restart sshd

rm -r ~/apt
sudo apt list '?narrow(?installed, ?not(?origin(Debian)))' | grep bullseye
sudo apt purge docker-scan-plugin

sudo reboot
```

And that's it! We've upgraded from Debian bullseye to trixie.

[Side note: The syntax highlighting is a little wonky here because I've customized it for Python which is most of the code on this site.]

I tried last but [it doesn't work anymore](https://www.debian.org/releases///trixie/release-notes/issues.en.html#the-last-lastb-and-lastlog-commands-have-been-replaced). I used this to get it working:
```bash
sudo apt install wtmpdb libpam-wtmpdb
echo "alias last='wtmpdb last'" >> ~/.bashrc
# SSH out and back in (or run .bashrc)
last # Works now
```