---
layout: project
title: "Linux Dual Boot Headless Debian"
date: 2023-01-14
cur_date: 2023-02-17
end_date: Present
categories: computing tinkering education
preview-image: linux.png
permalink: /projects/linux/
exclude-toc: true
---



{% capture description %}
In January 2023, I ordered a [refurbished PC](https://www.amazon.com/gp/product/B08BJDFZRF/ref=ppx_od_dt_b_asin_title_s00?ie=UTF8&psc=1) in order to install linux on it and tinker about with it.
{% endcapture %}
{% include proj-image-description.html image_url="/assets/images/linux-dual-boot-headless-debian/linux.png" %}

<br>

# Installing (Headless) Debian for Dual Boot on Windows 

* **Do not use Unetbootin. Just buy a flash drive. Debian fit on 512 MB flash drive.**

#### Paritioning (Windows)
* You need to pre-partition on the windows machine
* Disk Management > Right Click Windows (C:) > Shrink Volume
  * Mine is ~ 1000 GB so I want to split it in half for now.
  * There's 976144 MB total. I shrink by 524288 = 976144 - 451856 
  * So now there's 512 GB unallocated and that's where Debian will go.
* Control Panel > Hardware and Sound > Power Options > System Settings:
  * Uncheck: Turn on fast startup. This was recommended [here](https://www.debian.org/releases/bullseye/amd64/ch03s06.en.html#disable-fast-boot).
* Download Debian: https://www.debian.org/distrib/netinst
  * Click amd64 and this will initiate the iso download
  <img width="594" alt="image" src="https://user-images.githubusercontent.com/29719483/213880321-cd845ddd-16e2-436a-b086-b0eadedf03a1.png">
* I have a 512MB Flash Drive. Originally I wanted Ubuntu, but it was 3.7 GB for the latest LTS. So I just ordered 5 8 GB flash drives from Amazon (after nearly breaking my new computer with Unetbootin). But, later I figured out I actually want Debian. The iso is only 407 MB so I'm going to see if that works...
* Download [Balena Etcher](https://www.balena.io/etcher)
* Pick the ISO downloaded previously and select the 512 MB flash drive.
* It looks like it's working and successfully flashed... restarting...

#### Installing Linux
* It still fast restarted (which makes me angry). I'm doing a full shutdown... still did fast boot
* 3rd time I press F12 and select the flash drive from the UEFI boot menu
* I select the graphical installer
* US/America etc.
* Hostname: debian
* [Domain Name: .local](https://superuser.com/questions/889456/correct-domain-name-for-a-home-desktop-linux-machine)
* Create root password
* Enter full name
* Create user: `<user>` (switch with actual username)
* Create `<user>` password
* Pacific time
* Partitioning - Guided - use the largest continuous free space
* It selected the free space (but it was 549 GB?)
* No proxy
* Select SSH Server and Standard System Utilities
* ~~Select GNOME and Standard System Utilities~~ (Do headless)

#### First Boot In
* Remove USB and boot into the new system
* It will say: debian login:
* Type root
* Enter root password

* Make sure sshd is running: `systemctl status sshd`
* Get ip: `ip a`
* From another machine: ssh `<user>`@`<ip>`

#### Next Steps and SSH
* [Add my user to sudoers file:](https://unix.stackexchange.com/a/588996/272581)
    ```
    $ su -l
    # apt-get install sudo
    # adduser <user> sudo
    # logout
    ```

* Follow same instructions as [here](https://rileypeterson.github.io/raspberry-pi/ssh-setup.html), except where I note a difference.
* For ssh-keygen I save to `~/.ssh/id_rsa_debian`
* I make a passphrase (Debian SSH Passphrase)
* Apple Keychain: `ssh-add --apple-use-keychain ~/.ssh/id_rsa_debian`
* [Add another IdentityFile line](https://stackoverflow.com/a/42666864/8927098):
  * `  IdentityFile ~/.ssh/id_rsa_debian`
* Skip "Require sudo to require password (on Pi)" steps because this is enabled by default

* For setting up static IP (on debian): 
  * `sudo apt install net-tools`
  * `ip -c link show`
  * Note device that looks like `en...`
  * Make a copy of the current config
  * `sudo cp /etc/network/interfaces ~/`
  * `sudo nano /etc/network/interfaces`
  * Comment out everything under primary network interface:
    ```
    # The primary network interface
    # allow-hotplug <en...>
    # iface <en...> inet dhcp
    ```
  * Add the following
    ```
    # The primary network interface static ip
    auto <en...>
    iface <en...> inet static
      address 192.168.AAA.BBB
      netmask 255.255.255.0
      gateway 192.168.XX.YY
    ```
  * Address: `192.168.AAA.BBB` <-- What you want local IP address to be ( increment my pi one by 1 digit i.e. zzz + 1 )
  * Netmask: `255.255.255.0`
  * Gateway (?): result of `netstat -nr` (192.168.XX.YY) underneath Gateway
  * `sudo systemctl restart networking`
  * `sudo reboot`
  * You can now remove the backup (`cd ~ && rm interfaces`)
  * Continue with instructions

* On Mac `nano ~/.ssh/config` and add:
    ```
    Host debian
    Hostname 192.168.AAA.BBB
    User <user>
    Port 22
    ```
* Setup external ssh
* Deco Port Forwarding
  * Service Type: Custom
  * Service Name: SSH Debian
  * Internal IP: 192.168.xx.zzz
  * Internal Port: 22
  * External Port: `<some_port_that_is_not_22>`
  * Protocol: TCP
* Then to test (from Mac on VPN): `ssh -A -Y <user>@<home_ip_address> -p <some_port_that_is_not_22>`
* And if the password login is disabled correctly you should get:
  * Permission denied (publickey) if you try from a different machine (e.g. raspberry pi)
* I am now [disabling root login](https://unix.stackexchange.com/questions/383301/should-i-disable-the-root-account-on-my-debian-pc-for-security)
  * `sudo passwd -d root`
  * `sudo passwd -l root`
  * If you need to get into root mode it's: (`sudo su -`)
* `sudo nano /etc/ssh/sshd_config` add `PermitRootLogin no` and then `sudo systemctl restart sshd`
* Done :)

#### Other Notes
* If you see this when trying to ssh from Mac:
  ```
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
  ```
* Just remove the host from ~/.ssh/known_hosts

#### Additional Configuration
* Lynis
  * `sudo apt-get install lynis`
  * `sudo lynis audit system`
* Unattended Upgrades (https://wiki.debian.org/UnattendedUpgrades)
  * `sudo apt-get install unattended-upgrades apt-listchanges`
  * `sudo su -`
  * `echo unattended-upgrades unattended-upgrades/enable_auto_updates boolean true | debconf-set-selections`
  * `dpkg-reconfigure -f noninteractive unattended-upgrades`