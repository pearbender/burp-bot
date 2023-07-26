# Burp finding AI

(Currently unfinished)

This is an AI model designed to detect burps in a twitch stream, and a bot using that model to automatically send a `!burp` command to the streamer's chat.

# Training the model

## Collecting the training data

The data for training can be collected and generated semi-automatically, although the quality might be not particularly great. If you have the training data prepared, just put it into `burps-audio/` and `not-burps-audio/` as `wav` clips.

Automatic data downloading relies on the streamer to have a `!burp` command that the users use to count the burps. The downloader script gets audio clips from a section of the vod which precedes every such comment.

To download the data automatically follow the provided steps:

1. Use [twitch chat downloader](https://www.twitchchatdownloader.com/) to download twitch chat data of the vods you want to use for training. Put the data into the `chats` folder. It is expected that chat filenames are in the format `twitch-chat-1746376348.csv`.

2. (Optional) Change the quality of the downloaded vods, by changing the `VIDEO_QUALITY` variable in the `download_burps.sh` script (`audio_only` can't be used, by default the lowest working one is selected). 

3. Run:
    ```
    TWITCH_TOKEN=<your-token> ./download_burps.sh
    ```
    Replace `<your-token>` with twitch's `auth-token` from the browser cookies. This will parse all the chat data files, download the appropriate vods, and run `clip-burps.py` on all of them. Generated clips will then be converted to `wav` using `ffmpeg`. The generated data will be put into `burps-audio/` and `non-burps-audio/` folders. The cut original video clips of the burps will be put into `burps/parsed/` and `not-burps/parsed/` folders respectively.

4. (Optional) Manually filter and augment the training data as required, as `clip-burps.py` can somewhat often generate false-positives and also rarely false-negatives

## Training the model

There are several parameters that can be tweaked (apart from changing the training data) that can affect the training process. Those parameters can be found in the `train.py` script:

* `BATCH_SIZE`: How many clips to run training on at the same time, does not affect results, only the training speed. Tweak it down if your training device's memory runs out during training.

* `EPOCHS`: How many times to run the whole dataset through the network. More epochs means more time to train, and better results. But too many can make the model overfit to the training data, loosing accuracy.

* `TRAINING_VALIDATION_SPLIT`: Fraction of the data to use for training vs validation. Model will not take the validation data into account, which allows it to be a good metric of the model's performance.

To start the training run:
```
python3 train.py
```
The resulting model will be saved as `model.pt`. If `model.pt` exists already, **it will be overwritten**.

# Using the model

There are several ways to use the model once it's trained. All the scripts expect the model to exist and be names `model.pt`.

## Finding burps in an audio clip

You can use the model to find all the burps in an audio clip. It does that by splitting the audio clip into a series of overlapping segments of the appropriate length, and feeding each one into the model, saving those that were marked as a burp. A couple parameters can be tweaked to achieve better results:

* `SLICE_OVERLAP`: How many different clips should contain each point in an audio clip. If it's 1, the start of the next clip will be the end of the previous one. If it's 2, the start of the next clip would be half way into the previous one etc. More overlap means more clips will contain each burp, checking whether there are several consecutive clips will thus decrease the probability of a false positive, and will also allow to find the burp locations more precisely by looking at the clip intersections.

To find the burps, there are several steps:

1. Preparing the audio. Most audio formats are supported, but for the best results use the quality similar to that the training data. If you want to run the detection on vods downloaded with `download_burps.sh` you can use ffmpeg to extract the audio as `flac` with:
    ```
    ffmpeg -i vods/<vod-name>.mkv -q:a 0 -map a <audio-file-name>.flac
    ```

2. Preparing the template. Currently the script requires an example of a training clip to get the correct length and sample rate. You can use any clip from the training data in either `burps-audio/` or `not-burps-audio/`.

3. Running the script. Once you have a template and an audio file ready, you can start burp detection with:
    ```
    python find-burps-with-net.py burps-audio/<template-file>.wav <audio-file-name>.flac
    ```

The script will create two directories: 
* `burp-find-temp/` is used temporarily while the script is running, and can be safely deleted after it is done.
* `burp-found-temp/` will contain the detected burp clips. You can use the results to improve your training data, refer to [Using found burps](#using-found-burps)

## Improving training data

### Using found burps

It is possible to quickly generate good training data by using a trained model on vods to generate more clips, and manually labeling them.

A labeler app can also be used to simplify labeling.

*TODO: add it to repo, write docs* 

To label the clips, each of them should be moved to `burp-found-temp/burps` and `burp-found-temp/non-burps` respectively. After the new clips are labeled, they can be easily moved to the training data folders using:
```
python add-labeled-data-from-finder.py
```

### Filtering existing training data

The model can be used to search for false-positives and false-negatives in the training data itself.

By running:
```
python eval.py
```
The model will be used to label all the training data. False-positives and False-negatives will be displayed and also copied to `eval-temp/false-positives` and `eval-temp/false-negatives` respectively.

There is no automated way to proceed. The found clips can be manually checked for being actual false-positives and false-negatives or being an incorrectly labeled piece of training data. In case the clip is incorrectly labeled, it should be manually moved to the appropriate folder.