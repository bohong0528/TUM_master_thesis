#!/bin/bash

# Load parallel
source $(which env_parallel.bash)

# Load config
source configuration.sh

CMD="${SCRIPTS_DIRECTORY}/queue_job.sh"
# Set the maximum number of concurrently scheduled jobs. Note that
# this does **not** correspond to the maximum number of runs your job 
# comprises! You should not need to change this.
MAX_CONCURRENT_JOBS=100

########################################################################
start_time=$(date +%s)

env_parallel --delay 0.5 --joblog joblog.csv -j ${MAX_CONCURRENT_JOBS} --csv --colsep ',' --resume-failed --progress $CMD :::: "${RUN_LIST}"
state=$?

end_time=$(date +%s)

dur=$(( end_time - start_time ))

if [ -n ${EMAIL} ]; then
        cat <<-EOF | mail ${EMAIL} -s "[CLUSTER] ${JOB_NAME} completed!" -a "From:$(whoami)@master.osm.wi.tum.de"
                Completed job ${JOB_NAME} (${EXECUTABLE}) with status ${state}!
                Start: $(date --date=@${start_time} +%d.%m.%Y\ %H:%M:%S)
                End:  $(date --date=@${end_time} +%d.%m.%Y\ %H:%M:%S)
                Duration: $((dur/86400)) days and $(date --date=@${dur} -u +%H:%M:%S) hours/minutes/seconds.
                $(cut -f 7 joblog.csv | tail -n +2 | egrep "[^0]" | wc -l)/$(tail -n +2 joblog.csv | wc -l) runs returned a non-zero exit code.
EOF
fi

exit ${state}
