#!/bin/sh


cd "$('dirname' -- "${0}")"
cat './host.docker_run.txt' | tr '\n' ' ' > './host.docker_run_main.sh'
sh './host.docker_run_main.sh'
