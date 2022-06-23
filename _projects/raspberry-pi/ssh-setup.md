---
layout: pi-nav
permalink: /raspberry-pi/ssh-setup.html
---

I have a Mac so these are the instructions for that. These instructions enable you to access your raspberry pi from another computer connected to the **same** network. I am following the [remote access instructions here](https://www.raspberrypi.com/documentation/computers/remote-access.html#introduction-to-remote-access):
* Find the IP Address
  * Make sure it's powered on and connected to Wifi
  * Run `ping raspberrypi.local` (on host)
  * Note the IP Address
* Enable SSH on Raspberry Pi
  * Go to Preferences > Raspberry Pi Configuration > Interfaces > SSH (Enable)
  * On host: `ssh username@192.xxx.xxx.xxx` <-- IP Address noted from earlier
  * When asked about authenticity of host enter `yes`
  * After ssh'ing on host: `touch ~/Desktop/foo.txt`
  * You'll see a txt file pop up on the Raspberry Pi Desktop!
* Set up ssh-rsa key from host
  * Run `ssh-keygen`
  * `ssh-copy-id username@192.xxx.xxx.xxx`
  * Add to Apple Key Chain: `ssh-add --apple-use-keychain ~/.ssh/id_rsa`
  * Add the following to `~/.ssh/config` (This enables persistence after reboot on Mac):
    ```
    Host *
      UseKeychain yes
      AddKeysToAgent yes
      IdentityFile ~/.ssh/id_rsa
    ```
* [Additional Steps to Secure Raspberry Pi](https://www.raspberrypi.com/documentation/computers/configuration.html#securing-your-raspberry-pi)
  * Disable SSH Password Login to Pi (on Pi):
    ![image](https://user-images.githubusercontent.com/29719483/168674508-137f7513-416f-47d3-8688-cfe30b197e27.png)
  * Require `sudo` to require password (on Pi):
    ![image](https://user-images.githubusercontent.com/29719483/168674295-937d0b71-03b2-44c8-bd55-99d3eb9a1835.png)
* [Prevent `Wi-Fi is currently blocked by rfkill.` message](https://raspberrypi.stackexchange.com/a/123724)
* Set other locale and configuration options
* [iterm2](https://iterm2.com/) Quick SSH Profile
  ![image](/assets/images/raspberry-pi/iterm.png)
* Lastly I enabled auto-login under the Raspberry Pi preferences so that if it ever rebooted it wouldn't hang at the login
* Also, get [Xquartz](https://www.xquartz.org/) and make sure the GUI forwarding works (e.g. `geany &` (on Pi via SSH))
* Turn off display using `vcgencmd display_power 0`, on is `vcgencmd display_power 1` or after reboot it turns back on
* One thing I notice is that if the ethernet cable is unplugged from my raspberry pi, but it's still connected over the wireless the private IP address changes. I can still login via ssh (using new IP address): 
  
  `ssh -Y yourusername@192.xxx.xxx.yyy`

* Set-up Permanent IP Address
  * I'm following the instructions [here](https://raspberrypi-guide.github.io/networking/set-up-static-ip-address)
  * First run `netstat -nr`
  * `sudo nano /etc/dhcpcd.conf` add the following:
    ```
    # Raspberry Pi Setup
    interface eth0
    metric 300
    static ip_address=192.168.xx.yyy/24
    static routers=192.168.xx.z
    static domain_name_servers=192.168.xx.z

    interface wlan0
    metric 200
    ```
* Easy ssh (`ssh pi`)
  * `nano ~/.ssh/config` add the following:
    ```
    Host pi
      Hostname 192.168.xx.yyy
      User your_username
      Port 22
      ForwardX11 yes
      ForwardX11Trusted yes
    ```
* So now you can get there either way (I also set up a iterm profile)

* I also forgot my wifi network:
  ```
  sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
  # Delete network=....
  # Crtl + x, y
  ```