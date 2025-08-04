<<<<<<< HEAD
cd "$('dirname' -- "${0}")" ;  sudo -A  docker  run  --tty --interactive --rm  --ipc host  --mount 'type=tmpfs,destination=/data/TMPFS,tmpfs-size=137438953472' -v "$(realpath .):/data/input" -v "CACHE:/usr/local/cargo/registry" -v "CACHE:/root/.cache"  -p '0.0.0.0:8000:8000/tcp'  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 107374182400  "$('cat' './image_name.txt')"  '/data/input/start.sh' ; 
=======
cd "$('dirname' -- "${0}")" ;  podman  run  --tty --interactive --rm  --ipc host  --mount 'type=tmpfs,destination=/data/TMPFS,tmpfs-size=137438953472' -v "$(realpath .):/data/input" -v "CACHE:/usr/local/cargo/registry" -v "CACHE:/root/.cache"  -p '0.0.0.0:8000:8000/tcp'  --ulimit memlock=-1 --ulimit stack=67108864  "$('cat' './image_name.txt')"  '/data/input/start.sh' ; 
>>>>>>> main
