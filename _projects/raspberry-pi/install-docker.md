---
layout: pi-nav
permalink: /raspberry-pi/install-docker.html
---

From shell:
```
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER

docker run hello-world
```

Get docker compose as well:
```
pip3 install docker-compose
docker-compose --version
```