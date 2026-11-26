# Sourced by the queues after DATA is set: stage the dataset's image arrays on the instance's local NVMe for the queue's
# lifetime (fast_data.py; training reads a copy only when it is complete, so any failure just means reading $DATA as before)
# and delete them when the queue exits, normally, on an error or on Ctrl-C. Skipped when an outer queue already staged
# them (JIT_FAST_DATA set: the outer queue owns the copies and their cleanup), with FAST_DATA=0, or without the local disk.
# FAST_DIR overrides the location (default /opt/dlami/nvme/jit_fast).
if [ -z "${JIT_FAST_DATA:-}" ] && [ "${FAST_DATA:-1}" = 1 ]; then
    FAST_DIR=${FAST_DIR:-/opt/dlami/nvme/jit_fast}
    if [ -d "$(dirname "$FAST_DIR")" ] && [ -w "$(dirname "$FAST_DIR")" ]; then
        python fast_data.py stage --data_dir="$DATA" --fast_dir="$FAST_DIR"
        export JIT_FAST_DATA=$FAST_DIR
        trap 'python fast_data.py cleanup --fast_dir="$FAST_DIR"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
    else
        echo "fast data: no writable local disk at $(dirname "$FAST_DIR"); reading $DATA"
    fi
fi
