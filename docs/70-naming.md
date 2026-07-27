# Name & motto

## Recommendation

> # Parity
> ### **Prove it, then move it.**

**Full tagline (README subtitle, landing hero):**
*Open models are ready. Parity proves it — on your traffic, your hardware, your budget.*

---

## Why `clickllm` should go

Honest assessment of the working name:

| | |
|---|---|
| ✅ | Short, pronounceable, suggests "easy" |
| ❌ | **Mis-signals the category.** It sounds like a local model launcher — Ollama, LM Studio, llmfit. That is precisely the category [ADR-0001](adr/0001-migration-not-platform.md) decided *not* to compete in. The name would recruit the wrong users and invite the wrong comparison. |
| ❌ | `llm`-suffix space is saturated: LiteLLM, OpenLLM, llm-d, llmfit, AnythingLLM, LocalLLM. Zero recall advantage. |
| ❌ | "Click" sells convenience. We sell *evidence*. Convenience is the commodity half. |

The name should carry the thesis. `clickllm` carries the opposite one.

---

## Why `Parity`

**Parity is literally the question the product answers.** "Have we reached parity with GPT-5 on our workload?" is the sentence a staff engineer says in the meeting where this gets decided. Owning that word means owning the conversation.

- **One real word.** No portmanteau, no dropped vowel, no `.ai` gimmick.
- **Technically native.** Feature parity, parity bit, parity check — it reads as engineering, not marketing.
- **Names the output, not the mechanism.** Mechanisms (proxy, eval, router) get commoditized. The verdict doesn't.
- **Scales past the wedge.** Parity between models, between quantizations, between versions of your own prompts. Stage ⑦ (Guard) is parity monitoring. Nothing in the name expires.
- **Verbs cleanly.** "Run Parity against Kimi." "Parity says 96% on codegen."

**Risks, stated:** common English word — trademark and SEO will be contested, and the exact-match domain is likely taken. Mitigate with a qualified domain (`parity.dev`, `runparity.com`, `parityhq.com`) and a distinctive wordmark. Also note the Ethereum client of the same name — different category, largely dormant, but check the mark before filing. **Do not skip a trademark search.**

---

## Why "Prove it, then move it."

The whole product in four words, in order:

- **Prove it** → ② Distill + ④ Prove. The equivalence matrix.
- **then move it** → ⑤ Deploy + ⑥ Cut over. The quality-gated migration.

It works because it's *sequenced*. Most infra mottos assert a benefit ("ship faster", "AI, simplified"). This one describes a discipline, and implicitly indicts the alternative: everyone else asks you to move *without* proving. It rhymes, so it survives being repeated in a hallway.

---

## Shortlist considered

| Name | Motto | Verdict |
|---|---|---|
| **Parity** | Prove it, then move it. | ★ **recommended** — names the question, engineering-native, scales |
| **Ferry** | Safe passage off closed models. | strong runner-up. Warmer, very memorable, easier mark. Weaker on the *proof* half — sounds like transport, and transport is the commodity. |
| **Verdict** | The evidence to switch. | decisive and on-theme, but sounds legal/adversarial; slightly heavy for a dev tool |
| **Defect** | Defect from closed models. | most viral framing, best double meaning — **disqualified**: "defect" means *bug*, fatal collision for a product whose job is quality assurance |
| **Freehold** | Own your inference. | great for the enterprise/sovereignty story, obscure outside UK/property contexts |
| **Unhook** | Unhook from closed models. | clean verb, but purely negative framing — defines us by what we leave, not what we prove |
| **Crossover** | The day open caught up. | evocative but passive; names a moment, not a capability |
| ~~clickllm~~ | — | mis-signals category, saturated suffix, sells the commodity half |

---

## Voice

Three rules, derived from the product's own principles:

1. **Numbers over adjectives.** "Saves $2,530/mo on 89% of your traffic," never "dramatically reduces cost."
2. **Lead with the regret.** Publish where open models lose. The honest failure is what makes the wins credible — and it's the thing no vendor-authored comparison will ever do.
3. **Never oversell the switch.** "Move 70% and keep a closed fallback" is a *win*, and saying so is what separates us from every hosted-inference vendor's marketing.

## Alternate motto lines

Ranked, for landing pages, conference slides, and the one-liner in a cold email:

1. **Prove it, then move it.** — primary
2. Open models are ready. Prove it on your traffic.
3. Your traffic. Your benchmark. Your call.
4. Don't switch on a leaderboard.
5. Migration, with receipts.
