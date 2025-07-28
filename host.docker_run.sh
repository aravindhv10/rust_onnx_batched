#!/bin/sh
A1="$(realpath "${1}")"
A2="$(realpath "${2}")"
cd "$('dirname' -- "${0}")"
cat './host.docker_run.txt' | tr '\n' ' ' > './host.docker_run_main.sh'
sh './host.docker_run_main.sh' "${A1}" "${A2}"
