# VR2AR Converter 3

Convert your adult VR Videos into Passthrough AR Videos.

- Difference to https://github.com/michael-mueller-git/vr2ar-converter is other platform (docker) and used method. v1 and v2 have some mask merge jitter problems which i solved in v3 by using a custom  ArVideoWriter. Due to the diffrent programming languages in v1 vs v3 i can not backport this fix to v1. Therefore i recommend to use v3.
- Difference to https://github.com/michael-mueller-git/vr2ar-converter-v2 is this container uses more modern models for background removal ([MatAnyone](https://github.com/pq-yang/MatAnyone)). This repo therefore replaces v2.

Use the provided container and deploy on device with nvida gpu. Then use the buildin `grad.io` webui to convert your videos.

## Highlevel Usage

1. Create smal video chunks you want to convert to passthrough with no scene changes (i recommend chunks with length smaler than 3 minutes)
2. Process these chunks via the provided gradio webui
3. Wait for complete of process.
4. Download the result
5. Combine your chunks back into on video

## Features

- Persistent job artifacts: completed renders land in `/jobs/completed` on the shared `vr2ar` volume so they survive container restarts.
- Checkpointed processing: resumable mask generation is stored under `/app/process` (backed by the `process-cache` volume) for safe crash recovery.
- Multi-GPU queue manager: the UI container dispatches pending jobs from `/jobs/pending` to worker-specific folders in `/jobs/workers/<id>`.
- Worker role support: launch additional containers with `VR2AR_ROLE=worker` to consume dispatched jobs on extra GPUs.

## Multi-GPU Deployment

1. Start the UI/dispatcher as usual: `docker compose up -d` (or the CUDA2 variant).
2. For each additional GPU, start a worker container that shares the volumes and pins a device, for example:
   ```sh
   docker compose --profile worker up -d vr2ar-worker
   docker run -d \
     --name vr2ar-worker-0 \
     --gpus "device=0" \
     -e VR2AR_ROLE=worker \
     -e VR2AR_WORKER_ID=worker-0 \
     -v vr2ar:/jobs \
     -v process-cache:/app/process \
     ghcr.io/michael-mueller-git/vr2ar-converter-v3:latest
   ```
   Use a unique `VR2AR_WORKER_ID` (or rely on the container hostname) and adjust `--gpus` per worker.
3. The UI automatically detects idle workers and dispatches jobs; results appear in the download list once renders hit `/jobs/completed`.
4. To scale down, stop the worker containers; queued jobs remain in `/jobs/pending` for the next available worker.


