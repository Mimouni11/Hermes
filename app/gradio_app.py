"""Gradio bench for the call line.

Not a demo page — a bench. Every stage shows its working, so a turn that goes
wrong can be traced to the stage that did it rather than guessed at: the
quality gate's verdict and SNR, what Whisper actually heard and how confident
it was, which intent rule fired, what the FAQ retrieved and at what score, and
the reply Piper speaks.

Four tabs:

- **Tour d'appel** — an audio file or the microphone goes in, a spoken reply
  comes out, and every panel in between is visible.
- **Texte** — the same routing without speaking, for debugging retrieval.
- **Examens** — inspect exam identification, clarification and transfer.
- **Voix** — type a line and hear it, with the text expansion shown, because
  a reference code read as a word is a failure you can only hear.

It wires the real modules rather than `pipeline.run_turn`, which does not
exist yet. When WP6 lands, this switches to it and the panels stay.

    python app/gradio_app.py
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from hermes import (  # noqa: E402
    audio_quality,
    embeddings,
    escalation,
    exam_bot,
    faq_rag,
    intent_gate,
    lexicon,
    llm,
    stt,
    tts,
)
from hermes.config import get_settings  # noqa: E402
from hermes.intent_gate import Action, Intent  # noqa: E402

#: What the agent says when the intent is one the pipeline cannot serve yet.
NOT_BUILT_YET = {
    Intent.NEW_APPOINTMENT: (
        "J'ai bien compris que vous voulez un rendez-vous. La prise de "
        "rendez-vous arrive à l'étape suivante du projet (WP6)."
    ),
    Intent.EXISTING_APPOINTMENT: (
        "J'ai bien compris qu'il s'agit d'un rendez-vous existant. Cette "
        "branche arrive à l'étape suivante du projet (WP6)."
    ),
}


@dataclass
class Stage:
    """One stage's timing and its panel."""

    name: str
    seconds: float
    panel: str


@dataclass
class TurnReport:
    """Everything one turn produced, for the panels."""

    reply: str = ""
    audio: str | None = None
    stages: list[Stage] = field(default_factory=list)
    error: str = ""

    def add(self, name: str, seconds: float, panel: str) -> None:
        self.stages.append(Stage(name, seconds, panel))

    @property
    def total(self) -> float:
        return sum(s.seconds for s in self.stages)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _bar(value: float, maximum: float = 1.0, width: int = 18) -> str:
    filled = int(round(max(0.0, min(value / maximum, 1.0)) * width))
    return "█" * filled + "·" * (width - filled)


def render_gate(report: audio_quality.AudioReport) -> str:
    """The quality gate's panel."""
    verdict = "✅ ok" if report.is_acceptable else f"🔁 repeat — {report.reason}"
    return (
        f"**{verdict}**\n\n"
        f"| | |\n|---|---|\n"
        f"| SNR estimé | {report.snr_db:.1f} dB `{_bar(report.snr_db, 40)}` |\n"
        f"| durée | {report.duration_s:.2f} s |\n"
        f"| niveau RMS | {audio_quality.dbfs(report.rms):.1f} dBFS |\n"
        f"| crête | {report.peak:.3f} |\n"
        f"| saturation | {report.clipping_ratio:.2%} |\n"
        f"| parole | {report.speech_ratio:.0%} |\n\n"
        f"_seuil configuré : {get_settings().min_snr_db} dB_"
    )


def render_stt(result: Any) -> str:
    """What Whisper heard, and how sure it was."""
    settings = get_settings()
    confident = result.confidence >= settings.min_confidence
    mark = "✅" if confident else "⚠️"
    return (
        f"### « {result.text} »\n\n"
        f"| | |\n|---|---|\n"
        f"| confiance | {mark} {result.confidence:.3f} "
        f"`{_bar(result.confidence)}` |\n"
        f"| no_speech | {result.no_speech_prob:.3f} |\n"
        f"| audio | {result.duration_s:.2f} s |\n"
        f"| décodage | {result.latency_s:.2f} s "
        f"({result.latency_s / max(result.duration_s, 1e-6):.2f}× temps réel) |\n"
        f"| modèle | {settings.stt_model_size} / {settings.stt_device}"
        f"/{settings.stt_compute_type} |\n\n"
        f"_seuil de confiance : {settings.min_confidence}_"
    )


def render_repair(result: Any) -> str:
    """What the lexicon changed, and what it suspects.

    Shown even when it did nothing, because "the transcript was clean" is
    information: it tells you the next stage saw exactly what Whisper wrote.
    """
    if not result.changed and not result.safety_flags:
        return "**lexique** — rien à corriger, aucun signalement"

    lines = ["**lexique**\n"]
    if result.changed:
        lines.append(f"> {result.repaired}\n")
        lines.append("| corrigé | → | score | règle |\n|---|---|---|---|")
        for edit in result.edits:
            lines.append(
                f"| `{edit.original}` | `{edit.replacement}` | "
                f"{edit.score:.2f} | {edit.rule} |"
            )
    if result.safety_flags:
        lines.append("\n⚠️ **termes de sécurité suspectés**\n")
        lines.append("| terme | entendu | score | source |\n|---|---|---|---|")
        for flag in result.safety_flags:
            lines.append(
                f"| {flag.term} | `{flag.matched}` | {flag.score:.2f} | "
                f"{flag.source} |"
            )
    return "\n".join(lines)


def render_intent(decision: intent_gate.IntentDecision) -> str:
    """Which layer decided, and on what numbers."""
    icons = {
        Action.ROUTE: "➡️",
        Action.CLARIFY: "❓",
        Action.REPHRASE: "🔁",
        Action.DECLINE: "🚫",
        Action.TRANSFER: "📞",
    }
    # A rule that fires before retrieval leaves no similarity and no
    # probability. Printing 0.000 would read as "scored zero" rather than
    # "never scored", which is the opposite of what the bench is for.
    short_circuited = not decision.neighbours and not decision.probabilities
    unscored = "_non évalué — décidé avant le classifieur_"

    lines = [
        f"### {icons.get(decision.action, '')} `{decision.intent.value}`"
        f" — {decision.action.value}",
        f"\n**règle déclenchée :** `{decision.reason}`\n",
        f"| | |\n|---|---|",
    ]
    if short_circuited:
        lines.append(f"| similarité (portée) | {unscored} |")
        lines.append(f"| probabilité (intention) | {unscored} |")
    else:
        lines.append(
            f"| similarité (portée) | {decision.score:.3f} "
            f"`{_bar(decision.score)}` |"
        )
        lines.append(
            f"| probabilité (intention) | {decision.probability:.3f} "
            f"`{_bar(decision.probability)}` |"
        )
        lines.append(f"| marge | {decision.margin:.3f} |")

    if decision.probabilities:
        lines.append("\n**probabilités du classifieur**\n")
        lines.append("| intention | p |\n|---|---|")
        for intent, p in sorted(
            decision.probabilities.items(), key=lambda kv: -kv[1]
        ):
            flag = " ⚠️" if (
                intent is Intent.MEDICAL_QUESTION
                and p >= get_settings().medical_probability_floor
            ) else ""
            lines.append(f"| {intent.value}{flag} | {p:.3f} `{_bar(p)}` |")

    if decision.neighbours:
        lines.append("\n**exemples les plus proches**\n")
        lines.append("| sim | intention | exemple |\n|---|---|---|")
        for n in decision.neighbours[:5]:
            lines.append(f"| {n.score:.3f} | {n.intent.value} | {n.text[:58]} |")

    return "\n".join(lines)


def render_faq(hits: list[faq_rag.Retrieved], accepted: bool) -> str:
    """What the FAQ returned, and whether it cleared the threshold."""
    settings = get_settings()
    if not hits:
        return "_aucune recherche FAQ pour cette intention_"

    header = (
        f"**{'✅ réponse retenue' if accepted else '❌ sous le seuil — transfert'}**"
        f"  _(seuil {settings.faq_accept_threshold})_\n\n"
        "| score | entrée | question indexée |\n|---|---|---|"
    )
    rows = [
        f"| {h.score:.3f} `{_bar(h.score)}` | `{h.faq_id}` | {h.entry.question[:52]} |"
        for h in hits
    ]
    body = "\n".join([header, *rows])
    if accepted:
        body += f"\n\n> {hits[0].entry.answer}"
    return body


def render_timing(report: TurnReport) -> str:
    """Where the seconds went, against the brief's 5-second budget."""
    budget = 5.0
    lines = [
        f"### {report.total:.2f} s au total "
        f"{'✅' if report.total <= budget else '⚠️ au-dessus du budget'}"
        f" _(cible : {budget:g} s)_\n",
        "| étape | durée |\n|---|---|",
    ]
    for stage in report.stages:
        share = stage.seconds / max(report.total, 1e-6)
        lines.append(
            f"| {stage.name} | {stage.seconds:.2f} s `{_bar(share)}` "
            f"{share:.0%} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The turn
# --------------------------------------------------------------------------

def route_and_reply(text: str, report: TurnReport) -> tuple[str, str, str]:
    """Run the gate and the FAQ on a transcript, returning the panels.

    The server is **not** checked up front. The medical rules and the length
    guard decide before anything touches the index, and that independence is
    worth keeping: a pacemaker question must still reach a human when the
    vector store is down. So the call is made first and a connection failure
    is translated afterwards, instead of being pre-empted.
    """
    started = time.perf_counter()
    try:
        decision = intent_gate.classify(text)
    except Exception as exc:  # noqa: BLE001 — translate, do not swallow
        from hermes import vector_store

        if not vector_store.ping(force=True):
            settings = get_settings()
            raise RuntimeError(
                f"Qdrant ne répond pas sur {settings.qdrant_host}:"
                f"{settings.qdrant_port}. Démarrez-le :\n"
                "    docker compose up -d"
            ) from exc
        raise
    report.add("intention", time.perf_counter() - started, "")

    reply = intent_gate.reply_for(decision)
    hits: list[faq_rag.Retrieved] = []
    accepted = False

    if decision.action is Action.ROUTE and decision.intent is Intent.INFO:
        started = time.perf_counter()
        hits = faq_rag.search(text)
        report.add("FAQ", time.perf_counter() - started, "")

        started = time.perf_counter()
        spoken, source = faq_rag.spoken_answer(text)
        report.add(f"réponse ({source})", time.perf_counter() - started, "")
        accepted = spoken is not None
        reply = spoken or (
            "Je n'ai pas la réponse à cette question. Je vous passe "
            "quelqu'un qui pourra vous renseigner."
        )
    elif decision.action is Action.ROUTE:
        reply = NOT_BUILT_YET.get(decision.intent, "")

    return reply, render_intent(decision), render_faq(hits, accepted)


def on_audio(audio_path: str | None, enforce_gate: bool, speak: bool):
    """Run one full audio turn and fill every panel."""
    blank = ("", None, "", "", "", "")
    if not audio_path:
        return ("_déposez un fichier audio ou enregistrez un tour_",
                None, "", "", "", "")

    report = TurnReport()
    try:
        started = time.perf_counter()
        samples, rate = audio_quality.load_audio(audio_path)
        quality = audio_quality.check_samples(samples, rate)
        report.add("qualité", time.perf_counter() - started, "")
        gate_panel = render_gate(quality)

        if enforce_gate and not quality.is_acceptable:
            reply = audio_quality.retry_prompt(quality)
            audio = tts.synthesise(reply).path if speak else None
            return (
                f"**{reply}**\n\n_(le portier a arrêté le tour ici)_",
                str(audio) if audio else None,
                gate_panel,
                "_non transcrit : le portier a rejeté l'audio_",
                "", render_timing(report),
            )

        # load_model() caches by (backend, size). Constructing STT() here
        # instead would reload the weights on every single turn and bill the
        # load to the transcription timing — which is exactly what the first
        # run of this bench did: 10.3 s for 4.3 s of audio, against the 2.5 s
        # the evaluation measured.
        engine = stt.load_model()
        started = time.perf_counter()
        transcript = engine.transcribe_full(samples, rate)
        report.add("transcription", time.perf_counter() - started, "")

        # Repair sits between the transcriber and everything that reads it.
        # The WER figures are computed on `transcript.text`, never on this.
        started = time.perf_counter()
        fixed = lexicon.repair(transcript.text)
        report.add("lexique", time.perf_counter() - started, "")
        stt_panel = render_stt(transcript) + "\n\n" + render_repair(fixed)

        # A safety flag means a word was probably a safety term heard wrong.
        # It transfers, and the confirmation question explains why rather than
        # asserting the agent knows what was said.
        if fixed.safety_flags:
            reply = (
                lexicon.confirmation_question(fixed.safety_flags)
                + " Je vous passe quelqu'un pour vérifier."
            )
            audio = tts.synthesise(reply).path if speak else None
            return (
                f"**{reply}**",
                str(audio) if audio else None,
                gate_panel, stt_panel,
                "_arrêté par un signalement du lexique, avant le routage_",
                render_timing(report),
            )

        reply, intent_panel, faq_panel = route_and_reply(fixed.repaired, report)

        audio = None
        if speak and reply:
            started = time.perf_counter()
            audio = tts.synthesise(reply).path
            report.add("synthèse", time.perf_counter() - started, "")

        return (
            f"**{reply}**",
            str(audio) if audio else None,
            gate_panel, stt_panel, intent_panel + "\n\n---\n\n" + faq_panel,
            render_timing(report),
        )
    except Exception as exc:  # noqa: BLE001 — a bench shows the failure
        return (
            f"**Erreur : {type(exc).__name__}**\n\n```\n{exc}\n```\n\n"
            f"<details><summary>trace</summary>\n\n```\n"
            f"{traceback.format_exc()}\n```\n</details>",
            *blank[1:],
        )


def on_text(text: str, speak: bool):
    """Route a typed sentence, with no audio in."""
    if not (text or "").strip():
        return "_écrivez une phrase_", None, "", ""

    report = TurnReport()
    try:
        fixed = lexicon.repair(text)
        rule = escalation.check_transcript(text, fixed.repaired, fixed.safety_flags)
        reply, intent_panel, faq_panel = route_and_reply(fixed.repaired, report)
        intent_panel = render_repair(fixed) + "\n\n---\n\n" + intent_panel

        audio = None
        if speak and reply:
            started = time.perf_counter()
            audio = tts.synthesise(reply).path
            report.add("synthèse", time.perf_counter() - started, "")

        banner = (
            f"> ⚠️ règle médicale déterministe déclenchée : `{rule}`\n\n"
            if rule else ""
        )
        return (
            f"{banner}**{reply}**",
            str(audio) if audio else None,
            intent_panel + "\n\n---\n\n" + faq_panel,
            render_timing(report),
        )
    except Exception as exc:  # noqa: BLE001
        return (f"**Erreur : {type(exc).__name__}**\n\n```\n{exc}\n```",
                None, "", "")


def on_speak(text: str):
    """Synthesise a line and show what the voice was actually given."""
    if not (text or "").strip():
        return None, "_écrivez une phrase_"
    try:
        started = time.perf_counter()
        speech = tts.synthesise(text, use_cache=False)
        elapsed = time.perf_counter() - started
        return str(speech.path), (
            f"**écrit**\n\n```\n{text}\n```\n\n"
            f"**prononcé** _(après expansion)_\n\n```\n{speech.text}\n```\n\n"
            f"| | |\n|---|---|\n"
            f"| durée | {speech.duration_s:.2f} s |\n"
            f"| synthèse | {elapsed:.2f} s |\n"
            f"| moteur | {speech.backend} |\n"
            f"| fréquence | {speech.sample_rate} Hz |"
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"**Erreur : {type(exc).__name__}**\n\n```\n{exc}\n```"


def on_exam(text: str, use_llm: bool) -> tuple[str, list[list[str | float]], str, str]:
    """Show one exam decision, along with the evidence used by the matcher."""
    if not (text or "").strip():
        return "_Écrivez une demande d'examen._", [], "", ""

    started = time.perf_counter()
    try:
        index = exam_bot.get_index()
        result = exam_bot.identify(text, index=index, use_llm=use_llm)
    except RuntimeError as exc:
        return "**Impossible d'identifier l'examen.**", [], "", str(exc)

    elapsed = time.perf_counter() - started
    labels = {
        exam_bot.Status.MATCHED: "Examen identifié",
        exam_bot.Status.CLARIFY: "Précision nécessaire",
        exam_bot.Status.TRANSFER: "Transfert au secrétariat",
        exam_bot.Status.NOT_FOUND: "Examen introuvable",
    }
    summary = [f"### {labels[result.status]}"]
    if result.exam_id:
        summary.append(f"**{index.exams[result.exam_id].name}** (`{result.exam_id}`)")
    if result.question:
        summary.append(result.question)
    llm_status = "non sollicité" if use_llm else "désactivé"
    if result.used_llm:
        llm_status = "utilisé"
    summary.append(
        f"Motif `{result.reason.value}` · Score {result.score:.1f} · "
        f"Marge {result.margin:.1f} · LLM {llm_status} · {elapsed:.3f} s"
    )
    candidates = [
        [candidate.display, ", ".join(candidate.exam_ids), round(candidate.score, 1)]
        for candidate in result.candidates
    ]
    return "\n\n".join(summary), candidates, result.repaired, "\n".join(result.trace)


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------

EXAMPLES_TEXT = [
    "Je voudrais un rendez-vous pour une IRM du genou droit.",
    "Vous êtes ouverts jusqu'à quelle heure le samedi ?",
    "J'ai un pacemaker, je peux faire l'IRM ?",
    "Est-ce que mon résultat est normal ?",
    "Comment récupérer mes résultats ?",
    "Je veux parler à quelqu'un.",
    "Je veux m'inscrire à la salle de sport.",
    "Je dois être à jeun pour l'échographie abdominale ?",
]

EXAMPLES_TTS = [
    "C'est confirmé. Votre IRM est le 2026-11-26 à 14h30. "
    "Votre code de référence est K7M2PQ.",
    "Il y a beaucoup de bruit autour de vous. Pouvez-vous répéter ?",
    "Je ne peux pas répondre aux questions médicales. Je vous passe quelqu'un.",
]

EXAMPLES_EXAM = [
    "IRM du genou droit",
    "IRM du genou",
    "Radiographie du rachis cervical ou radiographie du rachis cervical dynamique",
    "Coroscanner",
    "Je dois passer une radio des poumons",
]


def build_ui() -> gr.Blocks:
    """Assemble the bench."""
    settings = get_settings()

    with gr.Blocks(title="hermes-voice — banc d'essai") as ui:
        gr.Markdown(
            "# hermes-voice — banc d'essai\n"
            "Chaque étape montre ce qu'elle a décidé : qualité audio, "
            "transcription, routage, FAQ, identification d’examen, réponse parlée."
        )

        with gr.Tab("Tour d'appel"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio du patient (fichier ou micro)",
                    )
                    enforce = gr.Checkbox(
                        value=True,
                        label="Appliquer le portier qualité "
                              "(sinon transcrire quand même)",
                    )
                    speak_audio = gr.Checkbox(value=True, label="Répondre en audio")
                    run_audio = gr.Button("Lancer le tour", variant="primary")
                    reply_audio = gr.Audio(label="Réponse de l'agent", type="filepath")
                    reply_text = gr.Markdown(label="Réponse")
                with gr.Column(scale=1):
                    timing_panel = gr.Markdown()
            with gr.Row():
                gate_panel = gr.Markdown(label="Qualité")
                stt_panel = gr.Markdown(label="Transcription")
            routing_panel = gr.Markdown(label="Routage")

            run_audio.click(
                on_audio,
                inputs=[audio_in, enforce, speak_audio],
                outputs=[reply_text, reply_audio, gate_panel, stt_panel,
                         routing_panel, timing_panel],
            )

        with gr.Tab("Texte"):
            with gr.Row():
                text_in = gr.Textbox(
                    label="Phrase du patient", lines=2,
                    placeholder="Je voudrais un rendez-vous pour une IRM…",
                )
            speak_text = gr.Checkbox(value=False, label="Répondre en audio")
            run_text = gr.Button("Router", variant="primary")
            gr.Examples(EXAMPLES_TEXT, inputs=text_in, label="Exemples")
            text_reply = gr.Markdown()
            text_audio = gr.Audio(label="Réponse de l'agent", type="filepath")
            text_routing = gr.Markdown()
            text_timing = gr.Markdown()

            run_text.click(
                on_text,
                inputs=[text_in, speak_text],
                outputs=[text_reply, text_audio, text_routing, text_timing],
            )

        with gr.Tab("Examens"):
            exam_in = gr.Textbox(
                label="Demande d’examen", lines=2, placeholder="IRM du genou droit",
            )
            exam_llm = gr.Checkbox(value=False, label="Autoriser le recours au LLM")
            exam_run = gr.Button("Identifier l’examen", variant="primary")
            gr.Examples(EXAMPLES_EXAM, inputs=exam_in, label="Exemples")
            exam_result = gr.Markdown()
            exam_candidates = gr.Dataframe(
                headers=["Examen candidat", "IDs du groupe", "Score"],
                datatype=["str", "str", "number"], value=[], interactive=False,
                label="Candidats classés",
            )
            repaired_exam = gr.Textbox(label="Texte après correction", interactive=False)
            with gr.Accordion("Trace de décision", open=False):
                exam_trace = gr.Textbox(show_label=False, lines=8, interactive=False)
            exam_run.click(
                on_exam,
                inputs=[exam_in, exam_llm],
                outputs=[exam_result, exam_candidates, repaired_exam, exam_trace],
                api_name="identify_exam",
            )

        with gr.Tab("Voix"):
            gr.Markdown(
                "Le texte est développé avant d'atteindre la voix : un code "
                "de référence est épelé, une heure est dite en toutes lettres. "
                "C'est la seule façon d'entendre si ça marche."
            )
            tts_in = gr.Textbox(label="Texte à prononcer", lines=3)
            run_tts = gr.Button("Prononcer", variant="primary")
            gr.Examples(EXAMPLES_TTS, inputs=tts_in, label="Exemples")
            tts_audio = gr.Audio(label="Sortie Piper", type="filepath")
            tts_panel = gr.Markdown()

            run_tts.click(on_speak, inputs=[tts_in], outputs=[tts_audio, tts_panel])

        gr.Markdown(
            f"---\n`{settings.stt_model_size}` {settings.stt_device}/"
            f"{settings.stt_compute_type} · SNR ≥ {settings.min_snr_db} dB · "
            f"τ intention {settings.intent_accept_threshold} · "
            f"τ FAQ {settings.faq_accept_threshold} · "
            f"plancher médical {settings.medical_probability_floor}"
        )
    return ui


def main() -> int:
    """Check what the bench needs, then launch it."""
    from hermes import vector_store

    settings = get_settings()
    print("hermes-voice bench")

    if not vector_store.ping():
        print("  ! Qdrant is not answering — run: docker compose up -d")
    else:
        print(f"  Qdrant: {vector_store.describe()}")

    model = settings.paths.models / settings.intent_model_file
    if not model.exists():
        print("  ! no intent classifier — run: "
              "uv run python scripts/train_intent_classifier.py")

    if tts.find_voice_file() is None:
        print(f"  ! no Piper voice in {settings.paths.models / 'piper'}")

    # Load Whisper before the first turn rather than during it. Otherwise the
    # first latency reading carries the model load and reads about four times
    # the real cost, which is worse than a slow start: it is a wrong number on
    # the panel that exists to show the number.
    # Everything that loads lazily, loaded now. Each of these has been
    # measured billing itself to the first caller's turn: Whisper ~2.4 s, the
    # google-genai import ~14 s, the sentence-transformer ~10 s. A panel that
    # reports latency has to report the turn, not the cold start.
    print(f"  loading {settings.stt_model_size} on {settings.stt_device}...")
    started = time.perf_counter()
    stt.load_model().model
    print(f"  ready in {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    embeddings.load_model()
    print(f"  embeddings ready in {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    if llm.warm():
        print(f"  LLM ready in {time.perf_counter() - started:.1f} s "
              f"({settings.gemini_model}, budget {settings.llm_timeout_s}s)")
    else:
        print("  ! LLM unavailable — the stored FAQ answers will be used")

    build_ui().launch(inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
