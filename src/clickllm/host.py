"""`clickllm host` — where to run a model this machine cannot.

[`launch`][clickllm.launch] refuses when the model does not fit the box in front
of you, and names this module as the next step. This is that step: rank the
places that *can* run it, cheapest first, free tiers included when the model
genuinely fits their limits — and generate a deploy artifact for the one you
pick.

## What this is not

It is not hosted inference, and it never becomes any (invariant 1). Nothing here
holds a credential, calls a provider API, or moves your traffic. It reads a
dated price registry off disk, does the same sizing arithmetic `fit` does, and
writes files you run yourself. The provider bills you directly and clickllm
never learns that you deployed.

**No credentials, ever.** Not prompted for, not read from the environment, not
written into an artifact. Where a token is genuinely required — gated weights —
the generated file passes it through from *your* environment and says so.

**No egress.** Planning is offline. Every price below came off a published page
on a recorded date, and that date is printed next to the number. Nothing
re-checks it for you, because a tool that silently phoned a vendor would break
the promise this repo is built on (NFR-2).

## Prices go stale — the design says so out loud

Every figure carries `source` and `checked`. A price we could not confirm is
`None` and renders as `?`, never a plausible-looking guess: someone will budget
against these numbers, and a wrong one is worse than an absent one. Vast.ai is
the honest example — its rates are set per-host by supply and demand, so no
figure can be quoted here at all, only the board to go and read.

Re-checking is a human job with a documented path: `Provider.source` is the URL,
`Provider.checked` is when it was last true. `demo()` fails if any entry lacks
either.

## Why an option can be missing

A provider that cannot run the model is *reported*, with the shortfall, rather
than dropped. That is the whole value of the free-tier question: "ZeroGPU is
free and has 96 GB, and your 671B model needs 340 GB" is an answer. Silence is
not. Google Colab is excluded for a different reason entirely — its terms
disallow serving — and saying which of the two applies is the point.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from . import fit
from .catalog import QUANT_BITS, ModelSpec
from .engines import Setting
from .fit import Fit, Placement
from .hardware_catalog import TP_SCALING_EFFICIENCY, Profile
from .hardware_catalog import get as profile_by_id

# Container images and the argv-slicing rule, borrowed from the Kubernetes
# emitter so a `docker-compose.yml` and a `Deployment` cannot drift apart.
from .k8s.reconcile import IMAGES as ENGINE_IMAGES
from .plan import Requirements, Workload
from .plan import plan as configure

__all__ = [
    "REGISTRY",
    "Artifact",
    "Excluded",
    "FreeTier",
    "HostOption",
    "Offer",
    "Provider",
    "Survey",
    "artifact",
    "demo",
    "options",
    "survey",
]

GB = 1024**3

#: The day every price in [`REGISTRY`] was read off its source page. One date
#: rather than one per row: they were checked in a single pass, and pretending
#: otherwise would dress up a batch job as continuous verification.
CHECKED = "2026-07-28"

#: Hardware offered by these providers that the main catalogue does not list.
#: Kept here rather than pushed into `hardware_catalog.PROFILES` because these
#: are rental shapes, not machines someone specifies a build around — move one
#: over the day `clickllm where` should rank it too.
#: Bandwidth is vendor-published and checked on the date below.
EXTRA_PROFILES: tuple[Profile, ...] = (
    # nvidia.com/en-us/data-center/a40 — 696 GB/s (checked 2026-07-28)
    Profile("a40", "NVIDIA A40 48 GB", "nvidia", 48, 696),
    # nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000
    # — 1792 GB/s (checked 2026-07-28). The best GB-per-dollar shape on the
    # market right now, and the card behind Hugging Face's free ZeroGPU tier.
    Profile("rtx-pro-6000", "NVIDIA RTX PRO 6000 Blackwell 96 GB", "nvidia", 96, 1792),
)


# --------------------------------------------------------------------------- #
# The registry — data, not branches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Offer:
    """One rentable shape at one provider.

    `profile_id` points at the hardware, so capacity and bandwidth are stated
    once, in the catalogue, and a price cannot quietly disagree with the physics
    it is priced against.
    """

    profile_id: str
    #: The provider's own name for this shape, as it appears on their price page.
    label: str
    #: USD per hour. `None` where the provider publishes no fixed rate — a
    #: marketplace, or a "contact sales" row. Renders as `?`, never a guess.
    usd_per_hour: float | None
    #: The provider's own identifier for this shape, as their API spells it.
    #: Empty when it was not confirmed — an artifact generator refuses rather
    #: than inventing one, because a wrong accelerator string fails at deploy.
    sku: str = ""


@dataclass(frozen=True, slots=True)
class FreeTier:
    """What a provider gives away, and every limit on it.

    The limits are the load-bearing half. "Free GPU" with the quota omitted is
    how someone ends up with a demo that dies mid-eval.
    """

    what: str
    limits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Provider:
    """Somewhere you can run a model, and the provenance of everything claimed."""

    id: str
    name: str
    #: Which artifact generator applies: `docker`, `modal`, `hf-space`, or
    #: `manual` where the deploy surface is a console this tool has not verified.
    artifact_kind: str
    offers: tuple[Offer, ...]
    #: URL the prices and limits were read from.
    source: str
    #: ISO date they were read. Displayed next to every number they produced.
    checked: str
    #: What happens on the first request after idle. Free and serverless tiers
    #: pay this repeatedly; a rented pod pays it once and then bills while idle.
    cold_start: str
    free: FreeTier | None = None
    #: Non-empty means this is never offered, and this is why. For a provider
    #: whose *terms* rule it out, which no amount of memory would fix.
    unavailable: str = ""
    #: Steps for a `manual` artifact — the deploy path stated rather than
    #: generated, where generating it would mean guessing an API shape.
    steps: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


#: Where a model can run, and what it costs. Every figure below was read off
#: `source` on `checked`; nothing is interpolated, estimated, or remembered.
#:
#: Adding a provider is adding an entry here. It must never mean editing the
#: solver — if a new provider needs new logic, that is a signal the shape of
#: this dataclass is wrong, not that the solver needs a branch.
REGISTRY: tuple[Provider, ...] = (
    Provider(
        id="hf-zerogpu",
        name="Hugging Face Spaces (ZeroGPU)",
        artifact_kind="hf-space",
        # ZeroGPU publishes two sizes: `large` is half a card at 48 GB, `xlarge`
        # is the full 96 GB. Only the full card is offered here, because the
        # bandwidth a half-card slice actually gets is not published, and the
        # tokens/sec roofline below would be a fabrication without it.
        offers=(Offer("rtx-pro-6000", "ZeroGPU xlarge (full RTX PRO 6000, 96 GB)", 0.0, "xlarge"),),
        source="https://huggingface.co/docs/hub/spaces-zerogpu",
        checked=CHECKED,
        cold_start=(
            "the GPU is attached per function call and released after, so a cold "
            "call pays the weight load every time the Space has been idle"
        ),
        free=FreeTier(
            what="genuinely free: no card, no credits, no hourly rate",
            limits=(
                "a free account in good standing (verified email, >30 days old) "
                "may host up to 2 ZeroGPU Spaces",
                "a daily GPU-seconds quota applies; the figure is not published, "
                "and PRO accounts get 8x whatever it is",
                "GPU is function-scoped, not a persistent server — this is a demo "
                "endpoint, not somewhere to point production traffic",
                "Spaces are public unless you make them private",
            ),
        ),
        notes=(
            "the only free option here that can actually serve a request. Treat it "
            "as a place to prove the model, not a place to run a business.",
        ),
    ),
    Provider(
        id="runpod",
        name="RunPod (Pods)",
        artifact_kind="docker",
        offers=(
            Offer("l4", "L4 24 GB", 0.39, "NVIDIA L4"),
            Offer("a40", "A40 48 GB", 0.44, "NVIDIA A40"),
            Offer("rtx-4090", "RTX 4090 24 GB", 0.69, "NVIDIA GeForce RTX 4090"),
            Offer("l40s", "L40S 48 GB", 0.99, "NVIDIA L40S"),
            Offer("rtx-5090", "RTX 5090 32 GB", 0.99, "NVIDIA GeForce RTX 5090"),
            Offer("a100-80", "A100 PCIe 80 GB", 1.39, "NVIDIA A100 80GB PCIe"),
            Offer("rtx-pro-6000", "RTX PRO 6000 96 GB", 1.99, "NVIDIA RTX PRO 6000"),
            Offer("h100", "H100 PCIe 80 GB", 2.89, "NVIDIA H100 PCIe"),
            Offer("h200", "H200 141 GB", 4.39, "NVIDIA H200"),
            Offer("b200", "B200 180 GB", 5.89, "NVIDIA B200"),
        ),
        source="https://www.runpod.io/pricing",
        checked=CHECKED,
        cold_start=(
            "a pod is always-on once started and bills while idle — cheapest per "
            "GPU-hour here, most expensive if you forget it is running"
        ),
        notes=(
            "the RTX PRO 6000 row is the cheapest 96 GB anywhere in this registry, "
            "at roughly half the price of an 80 GB H100.",
        ),
    ),
    Provider(
        id="modal",
        name="Modal",
        artifact_kind="modal",
        # Modal publishes per-second; these are x3600, exactly as published.
        offers=(
            Offer("l4", "L4 24 GB", 0.7992, "L4"),
            Offer("l40s", "L40S 48 GB", 1.9512, "L40S"),
            Offer("a100-40", "A100 40 GB", 2.0988, "A100-40GB"),
            Offer("a100-80", "A100 80 GB", 2.4984, "A100-80GB"),
            Offer("rtx-pro-6000", "RTX PRO 6000 96 GB", 3.0312, "RTX-PRO-6000"),
            Offer("h100", "H100 80 GB", 3.9492, "H100"),
            Offer("h200", "H200 141 GB", 4.5396, "H200"),
            Offer("b200", "B200 180 GB", 6.2496, "B200"),
        ),
        source="https://modal.com/pricing",
        checked=CHECKED,
        cold_start=(
            "scales to zero and bills by the second, so an idle endpoint costs "
            "nothing — the trade is that a cold container pays image pull plus "
            "weight load before the first token"
        ),
        free=FreeTier(
            what="$30/month of compute credit on the Starter plan, no card to start",
            limits=(
                "$30 is about 37 hours of an L4 or 7.5 hours of an H100 at the "
                "rates above — a real evaluation budget, not a production one",
                "10 concurrent GPU containers, 3 workspace seats",
                "credit does not roll over month to month",
            ),
        ),
        notes=(
            "the best fit for bursty or intermittent traffic: nothing is billed "
            "between requests, which no rented pod can say.",
        ),
    ),
    Provider(
        id="hf-endpoints",
        name="Hugging Face Inference Endpoints",
        artifact_kind="manual",
        offers=(
            Offer("l4", "1x L4 24 GB", 0.80),
            Offer("l40s", "1x L40S 48 GB", 1.80),
            Offer("a100-80", "1x A100 80 GB", 2.50),
            Offer("rtx-pro-6000", "1x RTX PRO 6000 96 GB", 2.75),
            Offer("h100", "1x H100 80 GB", 4.50),
            Offer("h200", "1x H200 141 GB", 5.00),
            Offer("b200", "1x B200 180 GB", 9.25),
            Offer("h100-x2", "2x H100 160 GB", 9.00),
            Offer("h100-x4", "4x H100 320 GB", 18.00),
            Offer("h200-x8", "8x H200 1128 GB", 40.00),
        ),
        source="https://huggingface.co/pricing",
        checked=CHECKED,
        cold_start=(
            "scale-to-zero is available per endpoint; a cold endpoint pays the "
            "weight load, a warm one bills continuously"
        ),
        steps=(
            "open https://endpoints.huggingface.co and choose 'New endpoint'",
            "select the model repo below, then the instance in the OPTION line",
            "set the container to vLLM and paste the arguments from ARGUMENTS below",
            "deploy. The endpoint URL and its token are yours — clickllm never sees them",
        ),
        notes=(
            "no generated config: the create-endpoint API's instance identifiers "
            "were not confirmed, and a wrong one fails at deploy time. The steps "
            "above are stated instead of guessed.",
            "the only multi-GPU shapes in this registry with published prices, "
            "which is what makes a 400B+ model reachable at all.",
        ),
    ),
    Provider(
        id="vast",
        name="Vast.ai",
        artifact_kind="docker",
        # Prices are set per-host by supply and demand. Not "unknown to us" —
        # genuinely not a single number, so it is not quoted as one.
        offers=(
            Offer("rtx-4090", "RTX 4090 24 GB", None),
            Offer("a100-80", "A100 80 GB", None),
            Offer("h100", "H100 80 GB", None),
            Offer("h200", "H200 141 GB", None),
        ),
        source="https://vast.ai/pricing",
        checked=CHECKED,
        cold_start="a marketplace instance; provisioning time depends on the host you pick",
        notes=(
            "usually the cheapest per GPU-hour of anything here, and the only one "
            "whose price this tool refuses to state — it moves per host, per hour. "
            "Read the live board before committing spend.",
            "interruptible instances are cheaper again and can be reclaimed "
            "mid-request, which is fine for evals and wrong for serving.",
        ),
    ),
    Provider(
        id="together",
        name="Together AI (Dedicated Inference)",
        artifact_kind="manual",
        offers=(
            Offer("h100", "HGX H100 80 GB", 5.49),
            Offer("b200", "HGX B200 180 GB", 8.99),
        ),
        source="https://www.together.ai/pricing",
        checked=CHECKED,
        cold_start="managed: you get an endpoint, not a machine, and they handle the load",
        steps=(
            "install their CLI and authenticate with your own account",
            "create a dedicated endpoint for the model repo below on the OPTION hardware",
            "see https://docs.together.ai for the current command — it is not "
            "reproduced here, because an unverified flag is a failed deploy",
        ),
        notes=(
            "roughly 1.9x RunPod's rate for the same H100. You are buying an "
            "operated endpoint, not a GPU — worth it or not depending on whether "
            "you wanted to run the engine yourself.",
        ),
    ),
    Provider(
        id="fireworks",
        name="Fireworks AI (On-demand deployments)",
        artifact_kind="manual",
        offers=(
            Offer("h100", "H100 80 GB", 7.00),
            Offer("h200", "H200 141 GB", 7.00),
            Offer("b200", "B200 180 GB", 10.00),
        ),
        source="https://fireworks.ai/pricing",
        checked=CHECKED,
        cold_start="managed and billed per GPU-second, with start-up time not charged",
        steps=(
            "install their CLI and authenticate with your own account",
            "create an on-demand deployment for the model repo below",
            "see https://docs.fireworks.ai for the current command",
        ),
        notes=(
            "the most expensive GPU-hour in this registry — 2.4x RunPod's H100. "
            "Their serverless per-token pricing is a different product and may "
            "well be cheaper for low volume; this row is the dedicated one.",
        ),
    ),
    Provider(
        id="colab",
        name="Google Colab (free tier)",
        artifact_kind="manual",
        offers=(),
        source="https://research.google.com/colaboratory/faq.html",
        checked=CHECKED,
        cold_start="n/a",
        free=FreeTier(
            what="free GPU access — for interactive notebook compute only",
            limits=("not a hosting product; see the exclusion",),
        ),
        unavailable=(
            "excluded by Colab's own terms, not by its memory: the free tier "
            "disallows 'file hosting, media serving, or other web service "
            "offerings not related to interactive compute with Colab', and "
            "terminates runtimes driven through a web UI rather than the notebook. "
            "A tunnelled inference endpoint is exactly the disallowed shape, so it "
            "is not offered here at any model size"
        ),
    ),
)


def _no_free_reason(provider: Provider) -> str:
    """Why `--free` ruled this provider out, with the price when there is one."""
    priced = [o.usd_per_hour for o in provider.offers if o.usd_per_hour is not None]
    if not provider.offers:
        return "lists no hardware shapes"
    if not priced:
        # An unpriced offer is not a free one. Saying "from $0.00/hr" here would
        # invent the number this repo refuses to invent (invariant 6).
        return f"no free tier — {provider.name} publishes no price for its shapes"
    return f"no free tier — every {provider.name} shape is paid, from ${min(priced):.2f}/hr"


def _gives_free_hardware(provider: Provider) -> bool:
    """Whether this provider hands out GPU time, as opposed to credit for it."""
    return any(o.usd_per_hour == 0.0 for o in provider.offers) or (
        provider.free is not None and not provider.offers
    )


def _profile(profile_id: str) -> Profile:
    """The hardware behind an offer. Local extras first, then the catalogue."""
    for p in EXTRA_PROFILES:
        if p.id == profile_id:
            return p
    return profile_by_id(profile_id)  # raises KeyError naming the alternatives


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HostOption:
    """One place that can actually run this model, and what it will cost."""

    provider: Provider
    offer: Offer
    #: The sizing answer, from the same solver `clickllm fit` uses. Carries the
    #: hourly rate too, so `$/Mtok` is the arithmetic `where` already audits.
    placement: Placement

    @property
    def fit(self) -> Fit:
        assert self.placement.fit is not None  # noqa: S101 — infeasible never gets here
        return self.placement.fit

    @property
    def free(self) -> bool:
        return self.offer.usd_per_hour == 0.0

    @property
    def usd_per_hour(self) -> float | None:
        return self.offer.usd_per_hour

    @property
    def usd_per_mtok(self) -> float | None:
        """USD per million output tokens at saturation. `None` when unpriced."""
        return self.placement.cost_per_mtok_usd

    @property
    def memory_gb(self) -> float:
        """Total GPU memory on this shape, nameplate."""
        return _profile(self.offer.profile_id).total_memory_gb

    @property
    def price_note(self) -> str:
        """Every price carries this. It is never printed without it."""
        if self.offer.usd_per_hour is None:
            return f"price not published — see {self.provider.source}"
        return f"as of {self.provider.checked}, {self.provider.source}"

    def render(self) -> str:
        f = self.fit
        hourly = "free" if self.free else _usd(self.usd_per_hour, "/hr")
        lines = [
            f"  {self.provider.name} · {self.offer.label}",
            f"    {f.model.name} @ {f.quant}"
            f"   {f.total_bytes / GB:.1f} GB of {f.usable_bytes / GB:.1f} GB usable",
            f"    {hourly}   {self.price_note}",
        ]
        if f.tokens_per_sec:
            lines.append(
                f"    ~{f.tokens_per_sec:.0f} tok/s   (roofline estimate, not measured)"
                + (f"   ~{_usd(self.usd_per_mtok, '/Mtok')}" if self.usd_per_mtok else "")
            )
        # This row is printed beside hosted providers' per-token prices, so a
        # self-hosting figure that is the optimistic end of a band has to say so
        # here rather than only in `fit --explain`. That is the comparison the
        # whole product turns on.
        spread = self.placement.moe_batch_optimism
        if spread and spread > 1.0:
            lines.append(
                f"    MoE: weight read held at the active count — optimistic end of a"
                f" band up to {spread:.1f}x wide at this batch"
            )
        lines.append(f"    cold start: {self.provider.cold_start}")
        if self.provider.free:
            lines += [f"    free tier: {self.provider.free.what}"]
            lines += [f"      · {limit}" for limit in self.provider.free.limits]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Excluded:
    """A provider that was considered and cannot be used, with the reason.

    Reported rather than dropped. "The free tier has 96 GB and you need 340" is
    an answer; a shorter list with no explanation is not.
    """

    provider_id: str
    provider_name: str
    reason: str
    #: True when free *hardware* was the thing ruled out — the question people
    #: actually asked, so it is surfaced separately. Not set for a provider whose
    #: free tier is credit against paid hardware: losing $30 of Modal credit to a
    #: model that needs an 8-GPU node is not "the free tier does not fit".
    was_free: bool = False


@dataclass(frozen=True, slots=True)
class Survey:
    """Everywhere this model could run, everywhere it could not, and why."""

    model: ModelSpec
    context: int
    concurrency: int
    options: tuple[HostOption, ...]
    excluded: tuple[Excluded, ...]

    @property
    def cheapest(self) -> HostOption | None:
        return self.options[0] if self.options else None

    @property
    def free_options(self) -> tuple[HostOption, ...]:
        return tuple(o for o in self.options if o.free)

    def render(self) -> str:
        m = self.model
        head = [
            f"  {m.name} · {m.params_b:g}B"
            + (f" ({m.active_b:g}B active, MoE)" if m.is_moe else "")
            + f" · {m.license}"
            + ("" if m.license_ok else " !"),
            f"  at {self.context:,} context, concurrency {self.concurrency}",
            "",
        ]
        if not self.options:
            head.append("  Nothing in the provider registry can run this model at that shape.\n")
        else:
            head += [
                f"  {'provider':<22}{'shape':<24}{'quant':<7}"
                f"{'total':>8}{'~tok/s':>8}{'$/hr':>9}{'$/Mtok':>9}",
                f"  {'-' * 87}",
            ]
            for o in self.options:
                f = o.fit
                tps = f"{f.tokens_per_sec:.0f}" if f.tokens_per_sec else "?"
                hr = "FREE" if o.free else _usd(o.usd_per_hour)
                mt = "0.00" if o.free else _usd(o.usd_per_mtok)
                head.append(
                    f"  {o.provider.name[:20]:<22}{o.offer.label[:22]:<24}{f.quant:<7}"
                    f"{f.total_bytes / GB:>7.1f}G{tps:>8}{hr:>9}{mt:>9}"
                )

        tail: list[str] = []
        if self.excluded:
            tail += ["", "  NOT AVAILABLE"]
            for e in self.excluded:
                mark = "FREE TIER — " if e.was_free else ""
                tail += [f"  {e.provider_name[:20]:<22}{mark}{e.reason}"]

        checked = sorted({o.provider.checked for o in self.options})
        tail += [
            "",
            f"  Prices read {', '.join(checked) if checked else CHECKED} from each provider's "
            "own page; every source URL is in the detail below.",
            "  They move. Re-read the source before committing spend — nothing here "
            "checks it for you.",
            "  ? = no published rate. $/Mtok assumes the machine is saturated "
            "single-stream, so real cost is higher.",
            "  Throughput figures are roofline estimates, not measurements.",
            "",
        ]
        return "\n".join(head + tail)


def _usd(value: float | None, suffix: str = "") -> str:
    """A price, or `?`. Never a plausible-looking guess (invariant 6)."""
    return "?" if value is None else f"${value:,.2f}{suffix}"


def _sized(model: ModelSpec, quant: str | None, hw, context: int, concurrency: int) -> Fit | None:
    """The best fit on this hardware, or `None` if it does not fit at all."""
    if quant is None:
        return fit.best_quant(model, hw, context, concurrency)
    f = fit.solve(model, quant, hw, context, concurrency)
    return f if f.feasible else None


def _why_not(
    model: ModelSpec, provider: Provider, quant: str | None, context: int, concurrency: int
) -> str:
    """Why a provider's *largest* shape still cannot run this model."""
    if not provider.offers:
        return "no hardware shapes recorded for this provider"
    biggest = max(provider.offers, key=lambda o: _profile(o.profile_id).total_memory_gb)
    hw = _profile(biggest.profile_id).to_hardware()
    q = quant or min(model.quants, key=lambda x: QUANT_BITS[x])
    probe = fit.solve(model, q, hw, context, concurrency)
    # `fit._shortfall` steps the unit down so a deficit under half a gigabyte
    # never renders as "short by 0 GB", which reads as a solver bug rather than
    # as the answer.
    short = fit._shortfall(probe.total_bytes - hw.usable_bytes)
    return (
        f"largest shape is {biggest.label} — {hw.usable_gb:,.0f} GB usable, and "
        f"{model.name} needs {probe.total_bytes / GB:,.0f} GB at {q} "
        f"({context:,} ctx x{concurrency}). Short by {short}"
    )


def survey(
    model: ModelSpec,
    *,
    quant: str | None = None,
    context: int = 32_768,
    concurrency: int = 1,
    free_only: bool = False,
) -> Survey:
    """Everywhere this model can run, cheapest first, with the rejections.

    Sizing is [`fit`][clickllm.fit]'s, not a second implementation: each offer's
    hardware profile becomes a `Hardware` and goes through the same solver that
    answers `clickllm fit`. A hosting recommendation that disagreed with the
    local sizing answer would be worse than none.

    Args:
        model: catalogue entry to place.
        quant: force a quantisation. Default is the best one that fits each
            shape, which is not the same everywhere — a 96 GB card may take 8-bit
            where a 48 GB one takes 4-bit, and that difference is the answer.
        context: context length to serve.
        concurrency: simultaneous in-flight requests.
        free_only: consider only shapes that cost nothing.

    Returns:
        A [`Survey`]. `options` is ranked cheapest-first, so a free tier that
        genuinely fits lands at the top; unpriced shapes sort last because they
        cannot be compared, not because they are expensive.
    """
    found: list[HostOption] = []
    excluded: list[Excluded] = []

    for provider in REGISTRY:
        if provider.unavailable:
            excluded.append(
                Excluded(
                    provider.id,
                    provider.name,
                    provider.unavailable,
                    _gives_free_hardware(provider),
                )
            )
            continue

        shapes = [o for o in provider.offers if not free_only or o.usd_per_hour == 0.0]
        if not shapes:
            # `continue` here dropped the provider from options AND excluded, so
            # `--free` silently shortened the list — the exact thing `Excluded`
            # exists to prevent. "RunPod has no free tier" is the answer to the
            # question; a shorter list with no explanation is not.
            excluded.append(
                Excluded(
                    provider.id,
                    provider.name,
                    _no_free_reason(provider),
                    _gives_free_hardware(provider),
                )
            )
            continue

        fitting = []
        for offer in shapes:
            hw = _profile(offer.profile_id).to_hardware()
            f = _sized(model, quant, hw, context, concurrency)
            if f is None:
                continue
            f = _rescored_for_parallelism(f, offer, model, quant, context, concurrency)
            if f is None:
                # It fits on paper across every device, and not on the one the
                # plan will actually use. Dropping it is the honest outcome —
                # quoting the aggregate figure beside a single-device command is
                # a price for hardware the artifact does not configure.
                continue
            fitting.append(
                HostOption(
                    provider,
                    offer,
                    Placement(offer.profile_id, offer.label, f, "", offer.usd_per_hour),
                )
            )

        if fitting:
            found.extend(fitting)
        else:
            excluded.append(
                Excluded(
                    provider.id,
                    provider.name,
                    _why_not(model, provider, quant, context, concurrency),
                    _gives_free_hardware(provider),
                )
            )

    # Cheapest first. Unpriced last — `?` cannot be compared to a number, and
    # sorting it as zero would put the one option we cannot cost at the top.
    found.sort(
        key=lambda o: (
            o.usd_per_hour is None,
            o.usd_per_hour if o.usd_per_hour is not None else 0.0,
            -(o.fit.tokens_per_sec or 0.0),
        )
    )
    return Survey(model, context, concurrency, tuple(found), tuple(excluded))


def options(
    model: ModelSpec,
    *,
    quant: str | None = None,
    context: int = 32_768,
    concurrency: int = 1,
    free_only: bool = False,
) -> list[HostOption]:
    """Just the runnable options, cheapest first. See [`survey`] for the rest."""
    return list(
        survey(
            model, quant=quant, context=context, concurrency=concurrency, free_only=free_only
        ).options
    )


def find(survey_result: Survey, provider_id: str) -> HostOption | None:
    """The cheapest option at one provider, or `None` if it has none."""
    return next((o for o in survey_result.options if o.provider.id == provider_id), None)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Artifact:
    """Files that deploy this option, and how to run them.

    Standalone by contract (NFR-4): nothing generated here imports clickllm,
    references it at runtime, or needs it installed. Delete this tool and the
    files still deploy.
    """

    provider_id: str
    #: `(filename, content)`, in the order they are worth reading.
    files: tuple[tuple[str, str], ...]
    #: What to run, in order. Never run for you — every one of these spends money.
    how: tuple[str, ...]

    def write(self, directory: str | Path) -> list[Path]:
        """Write every file into `directory`, creating it if needed."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, content in self.files:
            path = out / name
            path.write_text(content)
            written.append(path)
        return written


def _field(label: str, text: str, width: int = 68) -> list[str]:
    """One provenance field, wrapped under a hanging indent.

    Wrapped rather than truncated: the engine reasons run to a couple of
    sentences, and a header that scrolls sideways is a header nobody reads.
    """
    out: list[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    pad = " " * 11
    return [f"{label:<11}{out[0]}"] + [f"{pad}{rest}" for rest in out[1:]]


def _gap_lines(gaps: tuple[str, ...]) -> list[str]:
    """Header lines for what the engine could not express. Flat, not nested.

    `_field` returns a *list* of lines, so these must be flattened into the
    header. Spreading a generator of lists instead put a list where
    `"\n".join` expected a string, and broke every artifact this module makes.
    """
    if not gaps:
        return []
    out: list[str] = []
    for i, g in enumerate(gaps):
        out += _field("NOT SET" if i == 0 else "", g)
    out += _field(
        "",
        "A gap above means the planner asked for something this engine has no "
        "verified flag for. Where that is quantisation, serve a checkpoint "
        "already in that format — the size stated above assumes it.",
    )
    return out


def _header(
    option: HostOption,
    repo: str,
    engine: str,
    why: str,
    today: str,
    gaps: tuple[str, ...] = (),
) -> list[str]:
    """Provenance. What was chosen, why, when, and what could not be expressed.

    `gaps` is not decoration. This module used to destructure it as `_gaps`
    and drop it, which turned a refusal into silence: the quantisation intent
    came back Unsupported ("q8 is a precision, not a quantisation method"),
    the argv went out with no `--quantization`, and the header above still
    said the model fit. vLLM then loaded the bf16 base repo — 61 GB against
    43 GB usable — and OOMed on boot under a document promising it would not.

    `launch` and `box` both surface their gaps. This was the one caller that
    threw the answer away.
    """
    f = option.fit
    price = "free" if option.free else _usd(option.usd_per_hour, "/hr")
    speed = (
        f"~{f.tokens_per_sec:.0f} tok/s single-stream — roofline estimate, not measured"
        if f.tokens_per_sec
        else "? — no bandwidth figure for this shape, so no estimate is offered"
    )
    return [
        f"clickllm host — generated {today}",
        "",
        *_field("MODEL", f"{f.model.name} @ {f.quant}  ({repo})"),
        *_field(
            "",
            f"{f.total_bytes / GB:.1f} GB required of {f.usable_bytes / GB:.1f} GB "
            f"usable, at {f.context:,} ctx x{f.concurrency}",
        ),
        *_field("OPTION", f"{option.provider.name} · {option.offer.label} · {price}"),
        *_field("PRICE", option.price_note),
        *_field(
            "",
            "Prices and free-tier limits both move. Re-read that page before you "
            "commit spend; nothing in this file re-checks it.",
        ),
        *_field(
            "WHY",
            f"{f.quant} is the best precision that fits {option.offer.label}'s "
            f"{option.memory_gb:g} GB at this shape. clickllm host ranks every "
            f"provider cheapest-first; this is the one that was selected.",
        ),
        *_field("ENGINE", f"{engine} — {why}"),
        *_field("SPEED", speed),
        # Printed even though it undercuts the WHY line above, because the
        # alternative is a document that claims a precision the command does not
        # set. When quantisation is in here, the argv serves the base checkpoint
        # at full precision and the GB figure above is wrong.
        *_gap_lines(gaps),
        *_field(
            "STANDALONE",
            "this file has no clickllm dependency and deploys with it "
            "uninstalled. Everything below is the target platform's own config.",
        ),
        *_field(
            "SECRETS",
            "clickllm holds none and asks for none. Anything secret below is "
            "passed through from your own environment, by you.",
        ),
    ]


def _commented(lines: list[str], marker: str = "#") -> str:
    return "\n".join(f"{marker} {line}".rstrip() for line in lines)


def _rescored_for_parallelism(
    f: Fit, offer: Offer, model: ModelSpec, quant: str, context: int, concurrency: int
) -> Fit | None:
    """Re-cost a multi-device shape at the bandwidth it will really run at.

    `fit` sizes an N-device shape on *aggregate* memory and bandwidth, which is
    right for "can this hardware hold the model". But the planner then decides
    the parallelism, and when the model fits on one device it correctly sets
    `--tensor-parallel-size 1` — sharding anyway costs an all-reduce per layer
    for nothing.

    Those two correct decisions disagree, and this module prints the result as
    one number. Measured: Qwen3-32B at q8 on 4x H100 was quoted 261 tok/s off
    11,892 GB/s aggregate, while the emitted command pins it to a single device
    at 3,350 GB/s — 3.5x over, straight into the $/Mtok column, which is the one
    figure here somebody budgets against.

    So the money follows the command, not the catalogue: re-solve against the
    devices the plan will actually use.
    """
    profile = _profile(offer.profile_id)
    if profile.devices <= 1 or not f.tokens_per_sec:
        return f
    hw = profile.to_hardware()
    deployment = configure(
        hw,
        Requirements(Workload.INTERACTIVE, concurrency=concurrency, context=context),
        model,
        f.quant,
    )
    knob = deployment.get(Setting.TENSOR_PARALLEL)
    used = int(knob.value) if knob and isinstance(knob.value, int) else profile.devices
    if used >= profile.devices:
        return f
    # Scale the same way the catalogue does. `profile.bandwidth_gbps * used`
    # skips the sub-linear TP factor `to_hardware` applies, so it agrees only at
    # used == 1 — which is the only value the planner produces today, and
    # exactly the sort of accidental agreement that breaks the day it returns 2.
    narrowed = replace(
        hw,
        devices=used,
        bandwidth_gbps=profile.bandwidth_gbps * (1 + (used - 1) * TP_SCALING_EFFICIENCY),
        usable_bytes=int(hw.usable_bytes * used / profile.devices),
    )
    rescored = _sized(model, f.quant, narrowed, context, concurrency)
    # None means it does NOT fit the parallelism the plan will actually use.
    #
    # This used to `return rescored or f` — falling back to the aggregate Fit,
    # which is the exact number this function exists to correct. The comment
    # justifying it said the planner "was wrong to narrow it", which is true and
    # is not a reason to quote a figure describing hardware the emitted argv will
    # not use. `plan._tensor_parallel` decides TP=1 by comparing weights alone to
    # the per-device share; it never looks at KV or overhead, so at high context
    # x concurrency it narrows to one device that the full footprint cannot fit.
    # The row then carried N-device throughput and price beside a
    # `--tensor-parallel-size 1` command that will not start.
    #
    # Returning None drops the option, and the caller reports it as excluded with
    # its shortfall — this module's own rule: report, do not quietly serve a
    # number that cannot be delivered.
    return rescored


def _engine_command(
    option: HostOption, repo: str, context: int, concurrency: int
) -> tuple[str, str, list[str], tuple[str, ...]]:
    """(engine, why, argv, gaps) for this option, from the existing planner.

    Which engine suits which workload is `plan`'s decision and stays there. A
    second engine table here would drift, and only one of the two would be the
    one that ran.
    """
    hw = _profile(option.offer.profile_id).to_hardware()
    deployment = configure(
        hw,
        Requirements(Workload.INTERACTIVE, concurrency=concurrency, context=context),
        option.fit.model,
        option.fit.quant,
    )
    argv, gaps = deployment.command(repo)
    if not argv:
        raise ValueError(
            f"no verified flag dialect for {deployment.engine.value}, which the "
            f"planner chose for {option.offer.label}; emitting another engine's "
            f"flags would produce a config that cannot start"
        )
    return deployment.engine.value, deployment.engine_why, argv, gaps


def _docker(option: HostOption, head: list[str], argv: list[str], engine: str) -> Artifact:
    """A compose file and a `docker run`. What RunPod and Vast.ai both consume."""
    # `Engine` is a StrEnum, so the engine's own id keys the image table directly.
    image = ENGINE_IMAGES.get(engine)  # type: ignore[arg-type]
    if not image:
        raise ValueError(f"no container image recorded for engine {engine!r}")

    devices = _profile(option.offer.profile_id).devices
    # The whole argv, with the image's entrypoint overridden — not flags appended
    # to whatever it prepends. Same rule as the box and Kubernetes emitters, and
    # for the same reason: every image in the table prepends something different
    # (`vllm serve`, `exec "$@"`, nothing at all), so counting tokens to drop was
    # wrong for all of them.
    args = list(argv)
    compose = "\n".join(
        [
            _commented(head),
            "",
            "services:",
            "  inference:",
            f"    image: {image}",
            # The image's own entrypoint is reset, so `command` below is the
            # whole argv rather than flags appended to whatever it prepends.
            "    entrypoint: []",
            # Every arg is a *quoted* YAML string. Bare `32768` parses as an
            # integer and compose refuses a non-string in `command`; one of these
            # flags also carries JSON, which unquoted would parse as a mapping.
            # `json.dumps` is exactly YAML's double-quoted scalar form.
            "    command:",
            *[f"      - {json.dumps(a)}" for a in args],
            "    ports:",
            '      - "8000:8000"',
            "    # Gated repos need a token. It is taken from YOUR shell, never",
            "    # from a file clickllm wrote — this line names it and nothing more.",
            "    environment:",
            "      - HUGGING_FACE_HUB_TOKEN",
            "    # The engine downloads the weights on first start. Mounting the",
            "    # cache means a restart does not fetch tens of GB again.",
            "    volumes:",
            "      - ./hf-cache:/root/.cache/huggingface",
            "    # vLLM and SGLang use shared memory for their worker processes;",
            "    # the default 64 MB is not enough and fails as a cryptic hang.",
            "    ipc: host",
            "    deploy:",
            "      resources:",
            "        reservations:",
            "          devices:",
            "            - driver: nvidia",
            f"              count: {devices}",
            "              capabilities: [gpu]",
            "",
        ]
    )
    run = "\n".join(
        [
            "#!/usr/bin/env sh",
            _commented(head),
            "",
            "set -eu",
            "",
            "docker run --rm -it \\",
            f"  --gpus {'all' if devices > 1 else '1'} --ipc=host -p 8000:8000 \\",
            # Same reset as the compose file: the command below is the whole
            # argv, not flags appended to the image's own entrypoint.
            '  --entrypoint "" \\',
            '  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \\',
            '  -v "$PWD/hf-cache:/root/.cache/huggingface" \\',
            f"  {image} \\",
            *[f"  {shlex.quote(a)} \\" for a in args[:-1]],
            f"  {shlex.quote(args[-1])}" if args else "",
            "",
        ]
    )
    return Artifact(
        option.provider.id,
        (("docker-compose.yml", compose), ("run.sh", run)),
        (
            f"start an instance of this shape ({option.offer.label}) at "
            f"{option.provider.name} — {option.provider.source}",
            "copy this directory onto it, or paste run.sh's command into the "
            "provider's 'container start command' field",
            "docker compose up   (or `sh run.sh` for plain docker)",
            "curl http://localhost:8000/v1/models",
            "stop the instance when you are done — a pod bills while idle",
        ),
    )


def _modal(
    option: HostOption, repo: str, head: list[str], argv: list[str], engine: str
) -> Artifact:
    """A real Modal app module. `modal serve app.py` and it is live."""
    sku = option.offer.sku
    if not sku:
        raise ValueError(
            f"Modal's own accelerator name for {option.offer.label!r} is not "
            f"recorded in the registry, and guessing it would produce an app "
            f"that fails at deploy. Pick another shape with --provider/--on"
        )
    pip = "vllm" if engine.startswith("vllm") else "sglang[all]"
    app = "\n".join(
        [
            '"""' + f"{option.fit.model.name} on Modal, serving an OpenAI-compatible API.",
            "",
            *head,
            '"""',
            "",
            "import subprocess",
            "",
            "import modal",
            "",
            f'MODEL = "{repo}"',
            f'GPU = "{sku}"',
            "MINUTES = 60",
            "",
            "# Unpinned on purpose: pinning a version this file cannot verify would",
            "# be a number without a source. Pin it yourself once you have a build",
            "# that works — you will want to.",
            "image = (",
            '    modal.Image.debian_slim(python_version="3.12")',
            f'    .pip_install("{pip}", "huggingface_hub[hf_transfer]")',
            '    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})',
            ")",
            "",
            "# Weights land in a Volume so the second cold start does not re-download",
            "# tens of GB. This is the single biggest cold-start win on Modal.",
            'cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)',
            "",
            f'app = modal.App("clickllm-{option.fit.model.id}", image=image)',
            "",
            "",
            "@app.function(",
            "    gpu=GPU,",
            '    volumes={"/root/.cache/huggingface": cache},',
            "    timeout=60 * MINUTES,",
            "    # Scale to zero after idle: this is why Modal is cheap for bursty",
            "    # traffic and expensive for steady traffic.",
            "    scaledown_window=15 * MINUTES,",
            ")",
            f"@modal.concurrent(max_inputs={max(1, option.fit.concurrency)})",
            "@modal.web_server(port=8000, startup_timeout=30 * MINUTES)",
            "def serve() -> None:",
            "    subprocess.Popen(CMD)",
            "",
            "",
            "# The engine's own arguments. Nothing here wraps it.",
            "CMD = [",
            *[f"    {json.dumps(a)}," for a in argv],
            "]",
            "",
        ]
    )
    return Artifact(
        option.provider.id,
        (("modal_app.py", app),),
        (
            "pip install modal && modal setup   # authenticates with YOUR account",
            "modal serve modal_app.py           # ephemeral, live while it runs",
            "modal deploy modal_app.py          # persistent, scales to zero when idle",
            "the printed URL serves /v1 — an OpenAI-compatible endpoint",
        ),
    )


def _hf_space(option: HostOption, repo: str, head: list[str]) -> Artifact:
    """A ZeroGPU Space. The only free option here that serves a real request.

    Deliberately not vLLM: ZeroGPU attaches the card per function call and takes
    it back afterwards, so a persistent server process is the wrong shape and
    would be killed. `transformers` under `@spaces.GPU` is what the platform is.
    """
    model = option.fit.model
    app = "\n".join(
        [
            '"""' + f"{model.name} on Hugging Face ZeroGPU.",
            "",
            *head,
            "",
            "ZeroGPU attaches a GPU for the duration of a decorated call and",
            "releases it after, so this uses transformers rather than a persistent",
            "server: vLLM would be killed between requests.",
            '"""',
            "",
            "import gradio as gr",
            "import spaces",
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'MODEL = "{repo}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL)",
            "# NOTE: this loads the checkpoint at its published precision.",
            f"# clickllm sized this Space at {option.fit.quant}, which transformers cannot",
            "# reach without a quantisation backend (bitsandbytes), and that is not",
            "# verified on ZeroGPU — so the figure in the header is the sized one,",
            "# not what this file loads. Serve an already-quantised checkpoint to",
            "# match it.",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    MODEL,",
            '    dtype="auto",',
            '    device_map="cuda",',
            ")",
            "",
            "",
            "# duration is the GPU-seconds budget for one call; it is charged",
            "# against the daily quota whether or not it is used.",
            "@spaces.GPU(duration=120)",
            "def generate(prompt: str, max_new_tokens: int = 512) -> str:",
            '    messages = [{"role": "user", "content": prompt}]',
            "    ids = tokenizer.apply_chat_template(",
            "        messages, add_generation_prompt=True, return_tensors='pt'",
            "    ).to(model.device)",
            "    with torch.no_grad():",
            "        out = model.generate(ids, max_new_tokens=max_new_tokens)",
            "    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)",
            "",
            "",
            "demo = gr.Interface(",
            "    fn=generate,",
            '    inputs=[gr.Textbox(label="prompt"), gr.Slider(16, 2048, 512, step=16)],',
            "    outputs=gr.Textbox(),",
            f'    title="{model.name}",',
            ")",
            "",
            'if __name__ == "__main__":',
            "    demo.launch()",
            "",
        ]
    )
    readme = "\n".join(
        [
            "---",
            f"title: {model.name}",
            f"emoji: {'🤖'}",
            "colorFrom: gray",
            "colorTo: blue",
            "sdk: gradio",
            "app_file: app.py",
            "pinned: false",
            "---",
            "",
            "```",
            *head,
            "```",
            "",
            "## Deploying this",
            "",
            "1. Create a new Space at https://huggingface.co/new-space, SDK **Gradio**.",
            "2. In the Space's **Settings → Hardware**, choose **ZeroGPU**"
            f" ({option.offer.label}).",
            "3. Push `app.py`, `requirements.txt` and this file to the Space repo.",
            "",
            "## What free costs you",
            "",
            *[
                f"- {limit}"
                for limit in (option.provider.free.limits if option.provider.free else ())
            ],
            "",
            f"Source for those limits: {option.provider.source} "
            f"(checked {option.provider.checked}).",
            "",
            "## About the tok/s figure in the header",
            "",
            "That number is the card's memory-bandwidth roofline. `transformers`",
            "has no paged KV cache and no continuous batching, so this Space will",
            "not reach it — treat the estimate as the ceiling the hardware allows,",
            "not as what this code will do. Rent a GPU and run vLLM when the",
            "number matters.",
        ]
    )
    reqs = "\n".join(
        [
            _commented(head),
            "",
            "gradio",
            "spaces",
            "torch",
            "transformers",
            "accelerate",
            "",
        ]
    )
    return Artifact(
        option.provider.id,
        (("app.py", app), ("requirements.txt", reqs), ("README.md", readme)),
        (
            "create a Gradio Space at https://huggingface.co/new-space",
            "set Settings -> Hardware to ZeroGPU",
            "push these three files to the Space repo",
            "the Space URL is the endpoint; a cold call pays the weight load",
        ),
    )


def _manual(option: HostOption, repo: str, head: list[str], argv: list[str]) -> Artifact:
    """No generated config, and the reason why, plus the steps that do work.

    Emitting a config against an API whose field names were not confirmed is
    the failure mode `engines` exists to prevent: it looks like an answer and
    fails minutes later inside someone else's deploy.
    """
    body = "\n".join(
        [
            f"# {option.provider.name} — {option.fit.model.name}",
            "",
            "```",
            *head,
            "```",
            "",
            "## No generated config",
            "",
            "This provider's deploy surface is a console or an API whose field",
            "names this tool has not verified. A config built on a guessed field",
            "fails at deploy time, with their error message, minutes later — so",
            "the steps are stated instead.",
            "",
            "## Steps",
            "",
            *[f"{i}. {s}" for i, s in enumerate(option.provider.steps, 1)],
            "",
            "## Values you will need",
            "",
            f"- **model**: `{repo}`",
            f"- **hardware**: {option.offer.label}",
            f"- **price**: {_usd(option.usd_per_hour, '/hr')} — {option.price_note}",
            "",
            "## Arguments",
            "",
            "The engine's own flags, if the provider lets you supply them:",
            "",
            "```",
            " ".join(shlex.quote(a) for a in argv),
            "```",
            "",
            "## Notes",
            "",
            *[f"- {n}" for n in option.provider.notes],
        ]
    )
    return Artifact(option.provider.id, (("README.md", body),), tuple(option.provider.steps))


def artifact(
    option: HostOption,
    *,
    repo: str | None = None,
    context: int | None = None,
    concurrency: int | None = None,
    today: str | None = None,
) -> Artifact:
    """A standalone deploy artifact for one option.

    Args:
        option: the choice, from [`survey`] or [`options`].
        repo: weights to serve. Defaults to the catalogue's `repo`, which is
            `None` when unknown — never guessed, so this raises rather than
            emitting a config that points at a repo nobody confirmed exists.
        context: overrides the surveyed context. Defaults to the option's own.
        concurrency: same, for in-flight requests.
        today: the generation date stamped in the header. Injected so the test
            suite is not a function of the calendar.

    Returns:
        An [`Artifact`]. Nothing in it imports clickllm (NFR-4).

    Raises:
        ValueError: no repo is known for the model, the planner chose an engine
            with no verified flag dialect, or the provider's own identifier for
            the chosen shape was never confirmed.
    """
    model = option.fit.model
    weights = repo or model.repo
    if not weights:
        raise ValueError(
            f"no Hugging Face repo is known for {model.id!r}; the catalogue "
            f"leaves it absent rather than guessing. Add it with "
            f"`clickllm catalog-add <repo> --params-b {model.params_b:g} --network`, "
            f"or pass repo= explicitly"
        )

    ctx = context if context is not None else option.fit.context
    conc = concurrency if concurrency is not None else option.fit.concurrency
    stamp = today or date.today().isoformat()

    kind = option.provider.artifact_kind
    if kind == "hf-space":
        # ZeroGPU is function-scoped, so there is no server argv to plan.
        why = (
            "ZeroGPU attaches the GPU per call and takes it back after, so a "
            "persistent server process would be killed between requests"
        )
        return _hf_space(option, weights, _header(option, weights, "transformers", why, stamp))

    engine, why, argv, gaps = _engine_command(option, weights, ctx, conc)
    head = _header(option, weights, engine, why, stamp, gaps)
    if kind == "docker":
        return _docker(option, head, argv, engine)
    if kind == "modal":
        return _modal(option, weights, head, argv, engine)
    if kind == "manual":
        return _manual(option, weights, head, argv)
    raise ValueError(f"unknown artifact kind {kind!r} for provider {option.provider.id!r}")


def to_dict(s: Survey) -> dict:
    """The survey as JSON-able data, for `--json` and the MCP surface."""
    return {
        "model": s.model.id,
        "context": s.context,
        "concurrency": s.concurrency,
        "options": [
            {
                "provider": o.provider.id,
                "provider_name": o.provider.name,
                "shape": o.offer.label,
                "memory_gb": o.memory_gb,
                "quant": o.fit.quant,
                "total_gb": round(o.fit.total_bytes / GB, 1),
                "tokens_per_sec": round(o.fit.tokens_per_sec) if o.fit.tokens_per_sec else None,
                "usd_per_hour": o.usd_per_hour,
                "usd_per_mtok": round(o.usd_per_mtok, 2) if o.usd_per_mtok is not None else None,
                "moe_batch_optimism": (
                    round(o.placement.moe_batch_optimism, 1)
                    if o.placement.moe_batch_optimism
                    else None
                ),
                "free": o.free,
                "price_source": o.provider.source,
                "price_checked": o.provider.checked,
                "cold_start": o.provider.cold_start,
                "estimate": "roofline, not measured",
            }
            for o in s.options
        ],
        "excluded": [
            {"provider": e.provider_id, "free_tier": e.was_free, "reason": e.reason}
            for e in s.excluded
        ],
    }


def demo() -> None:
    """Self-check. Run with `python -m clickllm.host`. Touches no network."""
    import tempfile

    from . import catalog

    # Every entry is sourced and dated. A price with no provenance is the
    # failure this registry exists to prevent.
    for p in REGISTRY:
        assert p.source.startswith("https://"), p.id
        assert date.fromisoformat(p.checked) <= date(2100, 1, 1), p.id
        assert p.cold_start, p.id
        assert p.artifact_kind in {"docker", "modal", "hf-space", "manual"}, p.id
        for o in p.offers:
            _profile(o.profile_id)  # raises if the hardware is not described
            assert o.usd_per_hour is None or o.usd_per_hour >= 0.0, (p.id, o.label)
        if p.artifact_kind == "manual" and not p.unavailable:
            assert p.steps, f"{p.id}: a manual artifact is only honest with steps"

    qwen = catalog.get("qwen3-30b-a3b")
    s = survey(qwen, context=32_768, concurrency=4)
    assert s.options, "a 30B MoE must be hostable somewhere"

    # Cheapest first, with unpriced last — `?` cannot be compared to a number.
    priced = [o.usd_per_hour for o in s.options if o.usd_per_hour is not None]
    assert priced == sorted(priced), priced
    assert all(o.usd_per_hour is None for o in s.options[len(priced) :])

    # The free tier fits this model, so it is the first row.
    assert s.options[0].free and s.options[0].provider.id == "hf-zerogpu"
    assert s.free_options and s.options[0].usd_per_mtok == 0.0

    # Colab is excluded by its terms, at every model size — never by memory.
    colab = next(e for e in s.excluded if e.provider_id == "colab")
    assert "terms" in colab.reason and "interactive compute" in colab.reason
    assert colab.was_free

    # A model too big for the free card is excluded with the shortfall, not
    # silently dropped: "96 GB, and you need N" is the answer people came for.
    huge = survey(catalog.get("kimi-k3"), context=8192, concurrency=1)
    free_out = next(e for e in huge.excluded if e.provider_id == "hf-zerogpu")
    assert free_out.was_free and "Short by" in free_out.reason
    assert not any(o.provider.id == "hf-zerogpu" for o in huge.options)
    assert all(e.reason for e in huge.excluded), "every exclusion needs a reason"

    # --free-only keeps only the shapes that cost nothing.
    assert all(o.free for o in options(qwen, context=8192, free_only=True))

    # More context needs more memory, so it cannot open up more options.
    wide = survey(qwen, context=131_072, concurrency=8)
    assert len(wide.options) <= len(s.options)

    # Artifacts: standalone, stamped, and free of anything clickllm.
    with tempfile.TemporaryDirectory() as tmp:
        for provider_id in ("runpod", "modal", "hf-zerogpu", "hf-endpoints"):
            option = find(s, provider_id)
            assert option is not None, provider_id
            art = artifact(option, today="2026-07-28")
            for name, content in art.files:
                assert "clickllm host — generated 2026-07-28" in content, (provider_id, name)
                assert "import clickllm" not in content, (provider_id, name)
                assert "clickllm." not in content.replace("clickllm host", ""), (provider_id, name)
            assert art.how, provider_id
            written = art.write(Path(tmp) / provider_id)
            assert all(p.exists() for p in written)

        docker = artifact(find(s, "runpod"), today="2026-07-28")
        compose = dict(docker.files)["docker-compose.yml"]
        assert "vllm/vllm-openai" in compose or "sglang" in compose
        assert qwen.repo in compose
        assert "ipc: host" in compose

        modal_app = dict(artifact(find(s, "modal"), today="2026-07-28").files)["modal_app.py"]
        assert 'GPU = "' in modal_app and "modal.web_server" in modal_app

        space = dict(artifact(find(s, "hf-zerogpu"), today="2026-07-28").files)
        assert "@spaces.GPU" in space["app.py"]
        assert "sdk: gradio" in space["README.md"]

    # JSON carries the provenance of every price it states.
    blob = to_dict(s)
    assert all(row["price_source"] and row["price_checked"] for row in blob["options"])
    json.dumps(blob)  # must be serialisable

    top = s.options[0]
    print(f"{qwen.name}: {len(s.options)} options, {len(s.excluded)} excluded")
    print(
        f"cheapest: {top.provider.name} · {top.offer.label} · "
        + ("free" if top.free else _usd(top.usd_per_hour, "/hr"))
    )
    print("ok")


if __name__ == "__main__":
    demo()
