#!/bin/zsh
gcloud compute tpus execution-groups create \
  --name=node-1 \
  --zone=europe-west4-a \
  --tf-version=2.7.0 \
  --accelerator-type=v2-8
