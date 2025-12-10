#!/bin/bash

while true; do

  # cpu utilisation across pods
  # kubectl get hpa -n task05 >> scaling.log
  kubectl get vpa -n task05 >> scaling.log

  # kubectl get deploy -n task05 >> scaling.log

  # cpu utilisation for each pod
  kubectl top pods -n task05 >> scaling.log

  sleep 5
done

