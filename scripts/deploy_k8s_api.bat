@echo off
set IMAGE_TAG=%1

if "%IMAGE_TAG%"=="" (
    echo 사용법: scripts\deploy_k8s_api.bat 이미지태그
    exit /b 1
)

kubectl set image deployment/ny-taxi-api ny-taxi-api=ghcr.io/loafingcat/ny-taxi-api:%IMAGE_TAG%
kubectl rollout status deployment/ny-taxi-api