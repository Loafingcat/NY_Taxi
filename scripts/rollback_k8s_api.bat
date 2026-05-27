@echo off

kubectl rollout undo deployment/ny-taxi-api
kubectl rollout status deployment/ny-taxi-api