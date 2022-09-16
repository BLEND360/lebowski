#!/usr/bin/env bash
docker ps -qa | xargs -r docker stop
docker ps -qa | xargs -r docker rm
docker images --format '{{.Repository}} {{.ID}}' | grep -Ev '^(nvidia|endpoint)' | awk '{ print $2 }' | xargs -r docker rmi -f
docker volume ls -q | xargs -r docker volume rm
docker network ls --format '{{.Name}}' | grep -Ev '^(bridge|host|none)$' | xargs -r docker network rm
