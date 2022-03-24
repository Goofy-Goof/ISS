#!/bin/zsh
gcloud compute tpus execution-groups create \
  --name=iss-node \
  --zone=asia-east1-c \
  --disk-size=300 \
  --machine-type=n1-standard-16 \
  --tf-version=2.7.0 \
  --accelerator-type=v2-8
