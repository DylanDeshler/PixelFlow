#!/usr/bin/env bash
# Follow the training output of whichever queue step is running: tails the most recently written step log in
# runs/queue_logs (the queues' own queue_*.log files only hold start / end lines) and switches when another step starts
# writing. Steps waiting on the GPU lock have empty logs that stop changing, so the running step's log is the newest.
#
#   bash queues/follow.sh              # Ctrl-C to stop; the queues keep running
#   bash queues/follow.sh 50           # show the last 50 lines on each switch (default 20)
dir=runs/queue_logs
lines=${1:-20}
cur="" pid=""
trap '[ -n "$pid" ] && kill $pid 2>/dev/null; exit 0' INT TERM
[ -d "$dir" ] || { echo "no $dir here: run from the repo root (where the queues were started)"; exit 1; }
echo "watching $dir for the running step's log..."
while true; do
    new=$(ls -t "$dir"/*.log 2>/dev/null | grep -v '/queue_[^/]*$' | head -1) # skip the queues' own queue_*.log files
    if [ -n "$new" ] && [ "$new" != "$cur" ]; then
        [ -n "$pid" ] && kill $pid 2>/dev/null
        cur=$new
        echo "==> $cur"
        tail -n "$lines" -F "$cur" &
        pid=$!
    fi
    sleep 10
done
