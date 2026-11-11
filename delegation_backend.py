"""delegation_backend.py: overflow past the cap onto a second provider.

WHAT this is: a Hermes >= 0.22.5 ``agent.delegation_provider`` seam
implementation, name ``hermes-treeps``. For a ``delegate_task`` batch it
answers, per task, "which route does this child run on": the primary
delegation route (``dict(config)``, i.e. inherit, exactly what the host would
do without this backend) for the first ``capacity_overflow_after``
MiniMax-class children - counting the ones already live in the ledger - and
``{provider: capacity_overflow_provider, model: capacity_overflow_model,
label: 'overflow'}`` for the rest.

WHY the lazy import: today's deployed host is 0.21.7 and has no
``agent.delegation_provider`` module. This file must import cleanly there
(the plugin loads and the inline gate half still works), so the seam is
imported only inside :func:`register_provider`, which returns False and logs
which branch ran when the seam is missing. The plugin stays whole, half (2)
just stays inert.

Import discipline: same as capacity.py - stdlib plus ``from . import ...``
of sibling modules only, nothing from the host at module import time.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from . import capacity

logger = logging.getLogger(__name__)

#: The ``delegation.backend`` value this provider is selected by.
PROVIDER_NAME = "hermes-treeps"


class HermesTreepsDelegationProvider:
    """The route decider, as a plain mixin.

    WHY not a direct ``DelegationProvider`` subclass: the ABC lives in the
    host module that 0.21.7 does not have, and a module-level subclass would
    make the whole plugin unimportable there. ``register_provider`` attaches
    the ABC at registration time instead.
    """

    name = PROVIDER_NAME

    @property
    def summary(self) -> str:
        return "hermes-treeps capacity overflow: primary route up to the cap, then a second provider"

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        """The route for one task: primary, or overflow past the MiniMax slots.

        ``task`` carries ``goal``, ``context`` and ``index`` (0-based position
        in the batch). ``config`` is the host's ``delegation`` section; an
        empty default route means "inherit the parent's route", which is
        exactly today's behaviour without this backend.
        """
        from .settings import settings

        s = settings()
        index = _task_index(task)
        # WHY the transcripts and not the ledger: a ledger row cannot name the
        # child's route (subagent_start carries no model/provider), so a child
        # this backend already sent to the overflow provider would still count
        # as primary and poison the next batch. The host's manifests name the
        # route per delegation; capacity.live_from_logs reads them.
        live_minimax = capacity.live_from_logs(minimax_only=True)
        overflow_after = int(s["capacity_overflow_after"] or 4)
        overflow = capacity.should_overflow(index, live_minimax, overflow_after)

        if overflow:
            route = {
                "provider": s["capacity_overflow_provider"],
                "model": s["capacity_overflow_model"],
                "label": "overflow",
            }
        else:
            route = dict(config) if isinstance(config, Mapping) else {}
        # One INFO line per decision, with everything a human needs to audit
        # it: which child in the batch, which route it got, and why.
        logger.info(
            "hermes-treeps delegation backend: child %d -> provider=%s model=%s overflow=%s",
            index,
            route.get("provider", ""),
            route.get("model", ""),
            "yes" if overflow else "no",
        )
        return route


def _task_index(task: Mapping[str, Any]) -> int:
    """The 0-based batch position of ``task``; 0 when the host did not say."""
    try:
        return int((task or {}).get("index") or 0)
    except (TypeError, ValueError):
        return 0


def register_provider(ctx: Any) -> bool:
    """Attach the provider to the host's delegation seam; False when absent.

    Called from the plugin's ``register()``. On a host predating the seam
    (0.21.7 today) it returns False - the overflow half stays inert and the
    inline gate half keeps working - and logs which branch ran.
    """
    try:
        from agent.delegation_provider import DelegationProvider  # type: ignore
    except Exception:
        logger.info(
            "hermes-treeps delegation backend: agent.delegation_provider is absent "
            "(host predates the >=0.22.5 seam); overflow routing stays inert"
        )
        return False
    try:
        # Attach the ABC at registration time, not import time. The metaclass
        # (ABCMeta via ProviderBase) is taken from DelegationProvider itself
        # so instantiation still checks the abstract methods - route_task and
        # name are defined on the mixin above.
        provider_cls = type(DelegationProvider)(
            "HermesTreepsDelegationProvider",
            (HermesTreepsDelegationProvider, DelegationProvider),
            {},
        )
        provider = provider_cls()
        ctx.register_delegation_provider(provider)
        logger.info(
            "hermes-treeps delegation backend: registered %r with the delegation seam",
            provider.name,
        )
        return True
    except Exception:
        logger.warning(
            "hermes-treeps delegation backend: the seam exists but registration failed",
            exc_info=True,
        )
        return False
