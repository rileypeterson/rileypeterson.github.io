---
layout: pi-nav
permalink: /raspberry-pi/ordering-and-setup.html
---

<!-- # Ordering and Setup -->
* [Ordering and Setup](#ordering-and-setup)
  * [Pre-Ordering and the Long Wait](#pre-ordering-and-the-long-wait)
  * [Gathering the Rest](#gathering-the-rest)
    * [Total Cost](#total-cost)
* [Writing OS to MicroSD](#writing-os-to-micro-sd)
* [Connecting and Booting Up](#connecting-and-booting-up)


## Pre-Ordering and the Long Wait
If you go to the [Raspberry Pi Website](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) you can find ordering options after clicking "Buy Now". Everything was sold out :(. Fortunately, [Canakit](https://www.canakit.com/raspberry-pi-4-2gb.html?cid=usd&src=raspberrypi) gave the offer to pre-order it. I pre-ordered it on January 18th, 2022 and it said "Pre-order ships February 28th" if I remember correctly. It ended up shipping on April 1st, 2022. This was the biggest hassle of the whole process, but at last on April 4th, I had received the pi. It was \$45.00 (excluding shipping).

![pi-box](/assets/images/raspberry-pi/preview.jpg)

## Gathering the Rest
Now that I had the computer I needed a few more things to get it fully setup:
* [Raspberry Pi Keyboard](https://www.raspberrypi.com/products/raspberry-pi-keyboard-and-hub/) (Optional)
  * \$17.00 - I had an existing wireless keyboard, but thought this looked cool and I prefer it over my previous one. Had to re-map the key bindings to be like the Mac keyboard.

![Keyboard](https://user-images.githubusercontent.com/29719483/168495283-280e9b55-5b8a-4aec-8553-c9a007da8357.png){:class="rasp-image"}

* [Raspberry Pi Mouse](https://www.raspberrypi.com/products/raspberry-pi-mouse/) (Optional)
  * \$8.00 - I had an existing mouse, but thought this would be cool to have. I didn't really like it though, and don't use it currently. I bought the keyboard + mouse bundle when I ordered from [Canakit](https://www.canakit.com/official-raspberry-pi-keyboard-mouse.html?cid=usd&src=raspberrypi).

![Mouse](https://user-images.githubusercontent.com/29719483/168495372-6c995db5-4a2a-439e-9d34-6b436899fd59.png){:class="rasp-image"}

* [Raspberry Pi Case](https://www.amazon.com/dp/B07WCKLFLP) (Sort of Optional)
  * \$7.49 - Case was little finicky/cheap, but I'm glad I have it. 

![Case](https://m.media-amazon.com/images/I/81OKMOENoFL._AC_SX679_.jpg){:class="rasp-image"}

* [Raspberry Pi Power Supply](https://www.amazon.com/dp/B07W8XHMJZ)
  * \$7.99 - The power you need to turn on the raspberry pi.

![Power Supply](https://m.media-amazon.com/images/I/61pj7sQU3qL._AC_SX679_.jpg){:class="rasp-image"}

* [USB Mac Dongle](https://www.amazon.com/dp/B07S8MKJ6Q) (Optional, Mac Only)
  * \$36.98 - Needed this to write to the MicroSD card (came with adapter), but I also was meaning to get one of these for a while.

![Dongle](https://m.media-amazon.com/images/I/71JrzFDLxlL._AC_SX679_.jpg){:class="rasp-image"}

* [Micro HDMI to HDMI](https://www.amazon.com/dp/B06WWQ7KLV)
  * \$8.99 - Need this to display the screen, micro part goes into the raspberry pi and the full hdmi goes into a monitor (already had).

![HDMI](https://m.media-amazon.com/images/I/61tN4PIHfVL._AC_SY879_.jpg){:class="rasp-image"}

* [Micro SD Card 128 GB](https://www.amazon.com/dp/B07G3H5RBT)
  * \$24.81 - 128GB is a little over kill, but this is what you write the operating system onto and it stores all the data for your pi.

![SD Card](https://m.media-amazon.com/images/I/81P+FSZ40EL._AC_SX679_.jpg){:class="rasp-image"}


### Total Cost

In total excluding shipping it was \$156.26, but that includes optional things. 

You are probably looking at between \\$75-\$100 depending on what you already have at home.

## Writing OS to Micro SD
If you bought a [NOOBS Micro SD card](https://www.raspberrypi.com/news/introducing-noobs/), you don't need to do this step. Otherwise,
the raspberry pi needs an operating system to work. Following [these steps](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up/2), write the raspberry pi OS to the micro SD card. I used the adapter that came with the San Disk Micro SD and inserted that into the Mac dongle slot for the SD Card. I just did the recommended OS.

## Connecting and Booting Up
Basically following [these steps](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up/3):

1. Place into the case and insert the Micro SD card (with OS written to it) like so:
![Micro SD Insert](/assets/images/raspberry-pi/sd_card.jpg){:class="rasp-image-big"}
2. Connect ethernet cable (optional), mouse, keyboard, micro HDMI, and power supply (last):
![Connections](/assets/images/raspberry-pi/connections.jpg){:class="rasp-image-big"}
3. You should see the OS booting and the desktop appear on your monitor (connected via HDMI-Micro HDMI):
![Desktop](/assets/images/raspberry-pi/desktop.jpg){:class="rasp-image-big"}

Next we'll look at [setting up an SSH connection]({{ site.baseurl }}{% link _projects/raspberry-pi/ssh-setup.md %}) so that we can run the raspberry pi without needing the mouse, keyboard, or HDMI connections.