# VR2AR Converter 3

Convert your adult VR Videos into Passthrough AR Videos.

- Difference to https://github.com/michael-mueller-git/vr2ar-converter is other platform (docker) and used method. Depending on the video, v3 or the other is more suitable.
- Difference to https://github.com/michael-mueller-git/vr2ar-converter-v2 is this container uses more modern models for background removal ([MatAnyone](https://github.com/pq-yang/MatAnyone)). This repo therefore replaces v2.

Use the provided container and deploy on device with nvida gpu. Then use the buildin `grad.io` webui to convert your videos.

Note: gradio is a bit buggy; the displayed process sometimes loses connection, but the conversion process continues in the background. The processed data is lost if you close the browser tab. I should look for a better solution here....

A workaround for now is to serve filebrowser with shared `/tmp` so we can download the processed files from there if grad.io fails.
