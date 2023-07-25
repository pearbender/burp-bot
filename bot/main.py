# requires twitchio

import asyncio
from twitchio.ext import commands


class Bot(commands.Bot):
    periodic_task = None

    def __init__(self):
        # TODO(laxader): Add config file
        super().__init__(token='token', prefix='?', initial_channels=['perokichi_neet'])

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
        await ctx.send(f'Hello {ctx.author.name}!')


    @commands.command()
    async def start(self, ctx: commands.Context):
        if self.periodic_task != None:
            await ctx.send(f'already running')
            return
        
        # TODO(laxader): Move the listener starting to bot connection event
        # maybe add some sort of a stream start/stop detection

        self.periodic_task = asyncio.create_task(run_task(self))
        await ctx.send(f'started periodic runner')


    @commands.command()
    async def stop(self, ctx: commands.Context):
        if self.periodic_task == None:
            await ctx.send(f'already stopped')
            return
        
        self.periodic_task.cancel()
        self.periodic_task = None
        await ctx.send(f'stopped periodic runner')


async def run_task(bot: Bot):
    while True:
        if len(bot.connected_channels) <= 0:
            print("No channels connected")
            break;
        
        channel = bot.connected_channels[0]

        # TODO(laxader): Add realtime twitch audio clip reading here, 
        # and call model evaluation on those clips.

        await channel.send("stuff")
        await asyncio.sleep(3)


bot = Bot()
bot.run()