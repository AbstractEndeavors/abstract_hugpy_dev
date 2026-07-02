"""HugpyBot — discord UI arm of the abstract_hugpy central service."""
from __future__ import annotations

import logging

import discord
from discord.ext import commands, tasks

from . import config
from .hugpy_client import HugpyClient
from .prefs import PrefsStore

log = logging.getLogger(__name__)

COGS = (".cogs.chat", ".cogs.tools", ".cogs.ml", ".cogs.ops")


class HugpyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        if config.MEMBERS_INTENT:
            intents.members = True   # privileged — also needs the portal toggle
        super().__init__(command_prefix="!", intents=intents)
        self.hugpy = HugpyClient(config.HUGPY_BASE_URL)
        self.prefs = PrefsStore()
        self._bridged: set[str] = set()   # channel ids wired to a console session

    async def setup_hook(self) -> None:
        for cog in COGS:
            await self.load_extension(cog, package=__package__)
        if config.GUILD_ID:
            guild = discord.Object(id=config.GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        # Deliver console-queued messages from a model into its Discord target.
        self._outbox_poller.start()
        # Report the channels the bot can see so the console can offer them as a
        # dropdown when binding a model to a channel.
        self._channel_reporter.start()
        # Members too, but only when the privileged intent is enabled.
        if config.MEMBERS_INTENT:
            self._member_reporter.start()

    async def on_ready(self) -> None:
        log.info("logged in as %s (%s); hugpy central: %s",
                 self.user, self.user.id, config.HUGPY_BASE_URL)

    async def close(self) -> None:
        self._outbox_poller.cancel()
        self._channel_reporter.cancel()
        if config.MEMBERS_INTENT:
            self._member_reporter.cancel()
        await self.hugpy.close()
        await super().close()

    def model_for(self, user_id: int) -> str | None:
        """Local fallback: the user's saved pref, else the configured default."""
        return self.prefs.get_model(user_id) or config.DEFAULT_MODEL_KEY

    # ── F4 runtime settings (central is the SoT; the bot only reads) ──────
    _SETTINGS_TTL = 10.0

    async def settings_ns(self, ns: str) -> dict:
        """Cached view of a central settings namespace. Central unreachable →
        the last known values (or {}), so a blip never mutes the bot."""
        cache = getattr(self, "_settings_cache", None)
        if cache is None:
            cache = self._settings_cache = {}
        hit = cache.get(ns)
        import time as _time
        now = _time.monotonic()
        if hit and now - hit[0] < self._SETTINGS_TTL:
            return hit[1]
        try:
            values = await self.hugpy.settings_ns(ns)
        except Exception:
            log.debug("settings fetch failed for %s", ns, exc_info=True)
            values = hit[1] if hit else {}
        cache[ns] = (now, values)
        return values

    async def channel_settings(self, channel_id) -> dict:
        return (await self.settings_ns("discord.channels")).get(
            str(channel_id)) or {}

    async def channel_personality(self, channel_id) -> dict | None:
        """DISC-06: the personality assigned to this channel (from the
        personality registry), or None."""
        name = (await self.channel_settings(channel_id)).get("personality")
        if not name:
            return None
        p = (await self.settings_ns("personalities")).get(str(name))
        return {**p, "name": name} if isinstance(p, dict) else None

    async def set_user_model(self, user_id: int, model_key: str | None) -> None:
        """Write-through: central settings (authoritative, console-visible)
        plus the local prefs file (offline fallback)."""
        try:
            await self.hugpy.set_user_model(user_id, model_key)
        except Exception:
            log.debug("central pref write failed; local only", exc_info=True)
        await self.prefs.set_model(user_id, model_key)

    async def resolve_model_for(self, user_id: int, channel_id: int | None = None) -> str | None:
        """Pick the model for a turn. Precedence (defined once, here):
        explicit per-turn choice is handled by the caller before this runs;
        then console-managed Discord binding > channel personality's model
        (the personality owns its model) > channel delegation (CON-07,
        fallback when the personality doesn't pin one) > user pref (central
        settings, then local file) > configured default."""
        try:
            bound = await self.hugpy.resolve_discord_model(channel_id=channel_id, user_id=user_id)
            if bound:
                return bound
        except Exception:
            log.debug("discord resolve failed; trying settings", exc_info=True)
        if channel_id is not None:
            persona = await self.channel_personality(channel_id)
            if persona and persona.get("model_key"):
                return persona["model_key"]
            delegated = (await self.settings_ns("delegation")).get(
                f"channel:{channel_id}")
            if delegated:
                return delegated
        central_pref = (await self.settings_ns("discord.users")).get(
            str(user_id)) or {}
        if central_pref.get("model"):
            return central_pref["model"]
        return self.model_for(user_id)

    # ── outbound: model -> Discord (the "mobile arm") ─────────────────────
    @tasks.loop(seconds=8.0)
    async def _outbox_poller(self) -> None:
        try:
            self._bridged = await self.hugpy.list_bridged_channels()
        except Exception:
            pass  # keep the last known set if central blips
        try:
            messages = await self.hugpy.drain_discord_outbox()
        except Exception:
            return  # central unreachable this tick; try again next loop
        for m in messages:
            await self._deliver_outbound(m)

    def is_bridged(self, channel_id) -> bool:
        return str(channel_id) in self._bridged

    async def relay_inbound(self, message) -> None:
        """Forward a bridged-channel message up to central's bridge inbox."""
        try:
            author = getattr(message.author, "display_name", None) or str(message.author)
            await self.hugpy.relay_inbound(
                channel_id=message.channel.id, author=author,
                content=message.content or "",
            )
        except Exception:
            log.debug("relay_inbound failed", exc_info=True)

    @_outbox_poller.before_loop
    async def _before_outbox(self) -> None:
        await self.wait_until_ready()

    # ── report visible channels -> central (for the console dropdown) ─────
    @tasks.loop(seconds=60.0)
    async def _channel_reporter(self) -> None:
        channels: list[dict] = []
        for guild in self.guilds:
            me = guild.me
            for ch in guild.text_channels:
                try:
                    if me is not None and not ch.permissions_for(me).send_messages:
                        continue  # only offer channels the bot can actually post in
                except Exception:
                    pass
                channels.append({
                    "id": str(ch.id),
                    "name": ch.name,
                    "guild": guild.name,
                    "guild_id": str(guild.id),
                })
        try:
            await self.hugpy.report_discord_channels(channels)
        except Exception:
            log.debug("channel report failed; will retry next loop", exc_info=True)

    @_channel_reporter.before_loop
    async def _before_channel_report(self) -> None:
        await self.wait_until_ready()

    # ── report guild members -> central (user dropdown; needs members intent) ─
    @tasks.loop(seconds=300.0)
    async def _member_reporter(self) -> None:
        users: dict[str, dict] = {}
        for guild in self.guilds:
            for m in guild.members:
                if m.bot:
                    continue  # don't offer bots as binding targets
                users[str(m.id)] = {
                    "id": str(m.id),
                    "name": m.display_name or m.name,
                    "guild": guild.name,
                }
        try:
            await self.hugpy.report_discord_users(list(users.values()))
        except Exception:
            log.debug("member report failed; will retry next loop", exc_info=True)

    @_member_reporter.before_loop
    async def _before_member_report(self) -> None:
        await self.wait_until_ready()

    async def _deliver_outbound(self, m: dict) -> None:
        content = (m.get("content") or "").strip()
        if not content:
            return
        channel_id, user_id = m.get("channel_id"), m.get("user_id")
        try:
            if channel_id:
                channel = self.get_channel(int(channel_id)) or await self.fetch_channel(int(channel_id))
                await channel.send(content)
            elif user_id:
                user = self.get_user(int(user_id)) or await self.fetch_user(int(user_id))
                await user.send(content)
        except Exception:
            log.warning("could not deliver outbound discord message %s", m.get("id"), exc_info=True)
