bash config/fc.sh
sudo nmcli dev disconnect wlp0s20f3
jupyter-lab  --no-browser &

########################################################################################
# reset GPU
# cat /sys/kernel/debug/dri/1/amdgpu_gpu_recover
#
# watch -d -n 0.5 /opt/rocm/bin/rocm-smi

