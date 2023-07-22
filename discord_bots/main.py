import math
from datetime import datetime, timezone

from discord import Colour, Embed, Member, Message, Reaction
from discord.abc import User
from discord.ext.commands import CommandError, CommandNotFound, Context, UserInputError, CheckFailure
from discord.ext.commands.errors import CommandOnCooldown

from discord_bots.config import API_KEY, COMMAND_PREFIX, CONFIG_VALID, CHANNEL_ID, SEED_ADMIN_IDS
from discord_bots.log import define_default_logger, define_logger
from .bot import bot
from .models import CustomCommand, Player, QueuePlayer, QueueWaitlistPlayer, Session
from .tasks import (
    add_player_task,
    afk_timer_task,
    map_rotation_task,
    queue_waitlist_task,
    vote_passed_waitlist_task,
)

define_default_logger()
log = define_logger(__name__)


def create_seed_admins():
    with Session() as session:
        for seed_admin_id in SEED_ADMIN_IDS:
            player = session.query(Player).filter(Player.id == seed_admin_id).first()
            if player:
                player.is_admin = True
            else:
                session.add(
                    Player(
                        id=seed_admin_id,
                        is_admin=True,
                        name='AUTO_GENERATED_ADMIN',
                        last_activity_at=datetime.now(timezone.utc),
                    )
                )
        session.commit()


@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")

    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        log.error("Could not find text channel for the bot to live in. Shutting down bot.")
        await bot.logout()
        return
    guild = channel.guild
    if not guild:
        log.error("Text channel of the bot dies not belong to a guild (server). Shutting down bot.")
        await bot.logout()
        return

    add_player_task.start()
    afk_timer_task.start()
    map_rotation_task.start()
    queue_waitlist_task.start()
    vote_passed_waitlist_task.start()


@bot.event
async def on_command_error(ctx: Context, error: CommandError):
    if isinstance(error, CommandNotFound) or isinstance(error, CheckFailure):
        log.debug(f"[on_command_error] {error}")
    elif isinstance(error, CommandOnCooldown):
        def singular_or_plural(number, singular, plural):
            return singular if number == 1 else plural

        if error.retry_after > 3600:
            in_hours = int(error.retry_after // 3600)
            available_in = f"{in_hours} {singular_or_plural(in_hours, 'hour', 'hours')}"
        elif error.retry_after > 60:
            in_minutes = int(error.retry_after // 60)
            available_in = f"{in_minutes} {singular_or_plural(in_minutes, 'minute', 'minutes')}"
        else:
            in_seconds = math.ceil(error.retry_after)
            available_in = f"{in_seconds} {singular_or_plural(in_seconds, 'second', 'seconds')}"

        await ctx.channel.send(
            embed=Embed(
                description=f"Command {COMMAND_PREFIX}{ctx.command.name} on cooldown. Available in about {available_in}.",
                colour=Colour.red(),
            )
        )
    elif isinstance(error, UserInputError):
        if ctx.command.usage:
            await ctx.channel.send(
                embed=Embed(
                    description=f"Usage: {COMMAND_PREFIX}{ctx.command.name} {ctx.command.usage}",
                    colour=Colour.red(),
                )
            )
        else:
            await ctx.channel.send(
                embed=Embed(
                    description=f"Usage: {COMMAND_PREFIX}{ctx.command.name} {ctx.command.signature}",
                    colour=Colour.red(),
                )
            )
    else:
        if ctx.command:
            log.info(f"[on_command_error] command: {ctx.command.name}, type: {type(error).__name__}, {error}",
                     exc_info=error)
        else:
            log.info(f"[on_command_error] type: {type(error).__name__}, {error}", exc_info=error)


@bot.event
async def on_message(message: Message):
    if message.channel.id == CHANNEL_ID:
        with Session() as session:
            player: Player | None = (
                session.query(Player).filter(Player.id == message.author.id).first()
            )
            if player:
                player.last_activity_at = datetime.now(timezone.utc)
                if player.name != message.author.display_name:
                    player.name = message.author.display_name
            else:
                session.add(
                    Player(
                        id=message.author.id,
                        name=message.author.display_name,
                        last_activity_at=datetime.now(timezone.utc),
                    )
                )
            session.commit()
        await bot.process_commands(message)

        # Custom commands below
        if not message.content.startswith(COMMAND_PREFIX):
            return

        bot_commands = {command.name for command in bot.commands}
        command_name = message.content.split(" ")[0][1:]
        with Session() as session:
            if command_name not in bot_commands:
                custom_command: CustomCommand | None = (
                    session.query(CustomCommand)
                    .filter(CustomCommand.name == command_name)
                    .first()
                )
                if custom_command:
                    await message.channel.send(content=custom_command.output)


@bot.event
async def on_reaction_add(reaction: Reaction, user: User | Member):
    with Session() as session:
        player: Player | None = session.query(Player).filter(Player.id == user.id).first()
        if player:
            player.last_activity_at = datetime.now(timezone.utc)
            session.commit()
        else:
            session.add(
                Player(
                    id=reaction.message.author.id,
                    name=reaction.message.author.display_name,
                    last_activity_at=datetime.now(timezone.utc),
                )
            )


@bot.event
async def on_join(member: Member):
    with Session() as session:
        player = session.query(Player).filter(Player.id == member.id).first()
        if player:
            player.name = member.name
            session.commit()
        else:
            session.add(Player(id=member.id, name=member.name))
            session.commit()


@bot.event
async def on_leave(member: Member):
    with Session() as session:
        session.query(QueuePlayer).filter(QueuePlayer.player_id == member.id).delete()
        session.query(QueueWaitlistPlayer).filter(
            QueueWaitlistPlayer.player_id == member.id
        ).delete()
        session.commit()


def main():
    if not CONFIG_VALID:
        log.error("You must provide a valid config!")
        return

    create_seed_admins()
    bot.run(API_KEY)


if __name__ == "__main__":
    main()
