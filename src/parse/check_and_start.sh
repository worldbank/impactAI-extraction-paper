#!/bin/bash

# Define variables
PROJECT_ID="impactai-430615"
ZONE="us-central1-a"
INSTANCE_NAME="pdf-parser-vm"
JOB_NAME="start-parsing-vm-3hours"
LOCKFILE="/tmp/vm_start.lock"

# Check if another instance is running
if [ -e ${LOCKFILE} ] && kill -0 `cat ${LOCKFILE}` 2>/dev/null; then
    echo "Another instance is already running. Exiting..."
    exit 1
fi

# Create lock file
echo $$ > ${LOCKFILE}

# Make sure the lockfile is removed when we exit and when we receive a signal
trap "rm -f ${LOCKFILE}; exit" INT TERM EXIT

# Check the VM status
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --format="get(status)")

if [[ "$STATUS" == "RUNNING" ]]; then
    echo "The VM is already running. No action taken."
else
    echo "The VM is not running. Triggering scheduler job..."
    gcloud scheduler jobs run $JOB_NAME --location=us-central1
fi
