# requires twitchio
import yaml
import asyncio
from twitchio.ext import commands
from twitchrealtimehandler import TwitchAudioGrabber
import numpy as np
import librosa

from evaluator import *
from audio_buffer import *


CONFIG_FILE = './twitch-auth.yaml'
MODEL_DIR = './models'
TEMPLATE_FILE = './template.wav'
DETECTED_BURPS_DIR = './burps'

AUDIO_SECTION_LENGTH = 1
WINDOW_SIZE = 4
MIN_IN_WINDOW = 3

BURP_COOLDOWN = 5


with open(CONFIG_FILE, 'r') as f:
    CONFIG = yaml.safe_load(f)


if not os.path.exists(DETECTED_BURPS_DIR):
    os.makedirs(DETECTED_BURPS_DIR)


def get_template_size(template_file):
    audio_data, sr = librosa.load(template_file, sr=None, mono=False)
    return audio_data[0].size, sr


SIZE, SR = get_template_size(TEMPLATE_FILE)


print(f"Template size {SIZE} at sample rate {SR}")


def find_models():
    return [os.path.join(MODEL_DIR, file) for file in os.listdir(
        MODEL_DIR) if file.lower().endswith(".pt")]


class Bot(commands.Bot):
    periodic_task = None

    def __init__(self):
        super().__init__(token=CONFIG['token'], prefix='?', initial_channels=['perokichi_neet'])

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
        

    async def event_message(self, message):
        if message.echo:
            return
        
        print(message.content)
        await self.handle_commands(message)


    @commands.command()
    async def hello(self, ctx: commands.Context):
        if ctx.author.name != "alexlarex":
            return
        
        await ctx.send(f'[bot] Hello {ctx.author.name}!')


    @commands.command()
    async def bcstart(self, ctx: commands.Context):
        if ctx.author.name != "alexlarex":
            return
        
        if self.periodic_task != None:
            await ctx.send(f'[bot] already running')
            return
        
        # TODO(laxader): Move the listener starting to bot connection event
        # maybe add some sort of a stream start/stop detection

        self.periodic_task = asyncio.create_task(run_task(self))
        await ctx.send(f'[bot] started burp counter')


    @commands.command()
    async def bcstop(self, ctx: commands.Context):
        if ctx.author.name != "alexlarex":
            return
        
        if self.periodic_task == None:
            await ctx.send(f'[bot] already stopped')
            return
        
        self.periodic_task.cancel()
        self.periodic_task = None
        await ctx.send(f'[bot] stopped burp counter')


async def run_task(bot: Bot):
    print("Creating grabber")

    audio_grabber = TwitchAudioGrabber(
        twitch_url="https://www.twitch.tv/perokichi_neet",
        blocking=False,  # wait until a segment is available
        segment_length=AUDIO_SECTION_LENGTH,  # segment length in seconds
        rate=SR,  # sampling rate of the audio
        channels=2,  # number of channels
        dtype=np.float32  # quality of the audio could be [np.int16, np.int32, np.float32, np.float64]
        )

    audio_buffer = AudioBuffer(8 * 44100)
    evaluator = MultiModelEvaluator(find_models())

    floating_window = [0 for x in range(WINDOW_SIZE)]
    current_index = 0
    cooldown = 0

    print("Starting loop")

    while True:
        if len(bot.connected_channels) <= 0:
            print("No channels connected")
            break;
        
        channel = bot.connected_channels[0]

        audio = audio_grabber.grab()
        
        if audio is not None:
            print("grabbed audio")
            # audio = librosa.resample(audio, orig_sr=16000, target_sr=SR)
            # print("resampled audio") jL%+qU99LH=@5GA
            audio_buffer.add_clip(audio)
            print("added to buffer")
            file = audio_buffer.save_latest(SR, SIZE)
            print(f"Processing {file}")
            if file is not None:
                if evaluator.is_burp_in_file(file):
                    shutil.move(file, DETECTED_BURPS_DIR)
                    floating_window[current_index] = 1
                else:
                    os.remove(file)
                    floating_window[current_index] = 0
                
                current_index = (current_index + 1) % WINDOW_SIZE

                if sum(floating_window) >= MIN_IN_WINDOW:
                    if cooldown <= 0:
                        print("BURP DETECTED!!!!!")
                        cooldown = BURP_COOLDOWN
                        await channel.send("[bot] detected burp")
                        await asyncio.sleep(1)
                        await channel.send("!burrp")
                    else:
                        cooldown = BURP_COOLDOWN
        #else:
            #print("No audio awailable")

        if cooldown > 0:
            cooldown -= 0.25
        await asyncio.sleep(0.25)


bot = Bot()
bot.run()
