# VR2AR Converter 3

Convert your adult VR Videos into Passthrough AR Videos.

- Difference to https://github.com/michael-mueller-git/vr2ar-converter is other platform (docker) and used method. v1 and v2 have some mask merge jitter problems which i solved in v3 by using a custom  ArVideoWriter. Due to the diffrent programming languages in v1 vs v3 i can not backport this fix to v1. Therefore i recommend to use v3.
- Difference to https://github.com/michael-mueller-git/vr2ar-converter-v2 is this container uses more modern models for background removal ([MatAnyone](https://github.com/pq-yang/MatAnyone)). This repo therefore replaces v2.

Use the provided container and deploy on device with nvida gpu. Then use the buildin `grad.io` webui to convert your videos.

## Usage

1. Create smal video chunks you want to convert to passthrough with no scene changes (i recommend chunks with length smaler than 3 minutes)
2. Process these chunks via the provided gradio webui
3. Wait for complete of process.
4. Download the result
5. Combine your chunks back into on video

## TODO

- To improve the output we need to reverse track the masks for each provided mask and merge the output of these. Because MatAnyone Memory bank can only track features it have seen in the frames before. When the model has e.g. an tattoo which was not visible in the init frame we get holes in the predicted mask. When we use 2 init mask one for the  frame with the missing feature visible we could by feeding the frames in reverse also track such features. When we then merge both generated masks we should se a better output. The main problem is the required memory to hold these frames. An implementation requires further considerations...
