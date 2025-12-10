#!/bin/bash

while true; do
  echo "=== $(date -Iseconds) ===" >> scaling.log
  kubectl get hpa -n task05 >> scaling.log
  kubectl get deploy -n task05 >> scaling.log
  kubectl top pods -n task05 >> scaling.log  
  echo "" >> scaling.log
  sleep 10
done
