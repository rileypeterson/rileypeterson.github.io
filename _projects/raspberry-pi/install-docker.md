---
layout: pi-nav
permalink: /raspberry-pi/install-docker.html
---

From shell:
```
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Run the following from a fresh terminal
docker run hello-world
```

Get docker compose as well:
```
pip3 install docker-compose
docker compose version
# Docker Compose version v2.10.2
```