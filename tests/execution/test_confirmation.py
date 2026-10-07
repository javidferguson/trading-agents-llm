"""The human gate. **The one thing in this system with no off switch.**

Three properties are load-bearing and each one exists because its absence cost
something:

* a closed stdin **declines** -- the options scanner's gate was inverted and
  returned True regardless of the answer
* approval needs the **ticker**, not `y` -- a yes/no prompt is answered by
  muscle memory
* an **expired** proposal refuses before a prompt is drawn -- a warning inside
  a prompt is a warning a tired person approves anyway

Plus the one this file adds over the options engine: §4 marks ``dissent`` and
``invalidation`` REQUIRED on ``FinalDecision`` *"because both surface in the
confirmation prompt"*. If they stop appearing, that requirement has quietly
become decoration.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from research_desk.execution import confirmation as conf
from research_desk.models.orders import OrderPlan, Violation
from research_desk.models.state import FinalDecision

AS_OF = date(2026, 10, 7)
DISSENT = "a liquidity shock would hit this name harder than the index does"
INVALIDATION = "a close below the 200-day moving average at 389.20"


def decision(**overrides) -> FinalDecision:
    body = {
        "action": "BUY", "symbol": "TSM", "conviction": 0.65,
        "target_weight_pct": 4.5, "horizon_days": 45, "stop_loss_pct": 8.0,
        "rationale": "relative strength is improving against the sector",
        "dissent": DISSENT, "invalidation": INVALIDATION,
        "expires_at": datetime.now() + timedelta(hours=12),
    }
    body.update(overrides)
    return FinalDecision(**body)


def plan(quantity: int = 12, **overrides) -> OrderPlan:
    body = {
        "symbol": "TSM", "as_of": AS_OF,
        "action": "BUY" if quantity > 0 else "SELL",
        "quantity": quantity, "reference_price": 472.15,
        "estimated_notional": abs(quantity) * 472.15,
        "target_weight_pct": 4.34, "current_weight_pct": 2.08,
        "binding_cap": "requested",
    }
    body.update(overrides)
    return OrderPlan(**body)


def render(**kwargs) -> str:
    args = {"limit_price": 472.62, "stop_price": 434.81}
    args.update(kwargs)
    d = args.pop("decision", decision())
    p = args.pop("plan", plan())
    return conf.render_decision(d, p, **args)


# --------------------------------------------------------------------------- #
# §4's required fields have to actually appear
# --------------------------------------------------------------------------- #


def test_the_dissent_is_in_the_prompt() -> None:
    """*"A human sees this text immediately before approving the order, and it
    is the single most useful thing on that screen."*

    Until Stage 7 it was written to a file nobody read at the moment of
    decision.
    """
    text = render()
    assert DISSENT in " ".join(text.split())
    assert "STRONGEST ARGUMENT AGAINST" in text


def test_the_invalidation_is_in_the_prompt() -> None:
    text = render()
    assert INVALIDATION in " ".join(text.split())
    assert "WHAT WOULD PROVE IT WRONG" in text


def test_the_numbers_being_approved_are_the_numbers_that_will_be_sent() -> None:
    text = render()
    assert "472.62" in text            # the limit
    assert "434.81" in text            # the stop
    assert "12 share(s)" in text
    assert "TSM" in text


def test_a_fund_manager_adjustment_is_surfaced() -> None:
    """If the committee changed the proposal, the human should see that it was
    changed rather than only the result."""
    text = render(decision=decision(
        fund_manager_adjustment="Reduced target weight from 5.83% to 4.5%."
    ))
    assert "FUND MANAGER ADJUSTED" in text
    assert "5.83%" in text


def test_compliance_warnings_are_shown_with_a_hanging_indent() -> None:
    """One warning must look like one warning.

    Indenting every wrapped line with the bullet turned a single long message
    into three that read as three findings.
    """
    long_warning = (
        "earnings blackout could not be evaluated -- only annual filings "
        "(20-F); a foreign private issuer announces quarterly results on 6-K"
    )
    text = render(plan=plan(violations=[
        Violation(rule="earnings_blackout", severity="warn", message=long_warning)
    ]))
    assert "COMPLIANCE WARNINGS" in text
    bullets = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
    assert len(bullets) == 1, f"one warning rendered as {len(bullets)} bullets"


def test_a_degraded_run_says_so_loudly() -> None:
    text = render(decision=decision(degraded=True))
    assert "THIS RUN DEGRADED" in text


def test_a_buy_with_no_stop_says_the_risk_cap_assumed_one() -> None:
    """Stated rather than omitted. Sizing's ``cap_risk`` sized this position on
    the arithmetic that a stop bounds the loss; if there is no stop, the human
    is the only remaining control."""
    text = render(stop_price=None)
    assert "NONE" in text
    assert "risk cap that sized this assumed one" in text


def test_the_quote_source_is_shown() -> None:
    """A limit priced off the previous close behaves very differently from one
    off a live spread."""
    text = render(quote_source="PREVIOUS CLOSE -- the market is not quoting this now")
    assert "PREVIOUS CLOSE" in text


def test_the_preflight_numbers_appear() -> None:
    text = render(preflight=conf.Preflight(
        init_margin_change="2832.90", maint_margin_change="1416.45",
        commission=1.0, commission_currency="USD", warning="price is outside the band",
    ))
    assert "2,832.90" in text or "2832.90" in text
    assert "1.00 USD" in text
    assert "price is outside the band" in text


def test_an_empty_preflight_is_omitted_rather_than_shown_as_n_a_everywhere() -> None:
    text = render(preflight=conf.Preflight())
    assert "BROKER PREFLIGHT" not in text


# --------------------------------------------------------------------------- #
# Expiry refuses, and refuses BEFORE prompting
# --------------------------------------------------------------------------- #


def test_an_expired_proposal_raises_rather_than_rendering() -> None:
    """*"A proposal generated after Tuesday's close must not be executable on
    Thursday -- render_decision() refuses outright rather than prompting."*"""
    with pytest.raises(conf.ExpiredProposalError, match="expired"):
        render(decision=decision(expires_at=datetime.now() - timedelta(hours=1)))


def test_the_expiry_message_says_how_stale_it_is() -> None:
    with pytest.raises(conf.ExpiredProposalError) as exc:
        conf.assert_not_expired(
            decision(expires_at=datetime(2026, 10, 1, 12, 0)),
            now=datetime(2026, 10, 3, 12, 0),
        )
    assert "48.0 hours ago" in str(exc.value)
    assert "desk decide" in str(exc.value)


def test_a_proposal_expiring_in_a_second_is_still_valid() -> None:
    conf.assert_not_expired(
        decision(expires_at=datetime(2026, 10, 7, 12, 0, 1)),
        now=datetime(2026, 10, 7, 12, 0, 0),
    )


def test_the_gate_never_prompts_for_an_expired_proposal(monkeypatch) -> None:
    """The important half: no ``input()`` is reached, so there is nothing for a
    tired person to approve."""
    def explode(*args, **kwargs):
        raise AssertionError("input() was called for an expired proposal")

    monkeypatch.setattr("builtins.input", explode)
    with pytest.raises(conf.ExpiredProposalError):
        conf.CLIConfirmationGate().confirm(
            decision(expires_at=datetime.now() - timedelta(hours=1)),
            plan(), limit_price=472.62,
        )


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #


def test_typing_the_ticker_approves(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "tsm")
    assert conf.CLIConfirmationGate().confirm(
        decision(), plan(), limit_price=472.62, stop_price=434.81
    ) is True


@pytest.mark.parametrize("answer", ["y", "yes", "Y", "", "NVDA", "ok", "12"])
def test_anything_other_than_the_ticker_declines(monkeypatch, answer: str) -> None:
    """*"A yes/no prompt is answered by muscle memory; a symbol has to be read
    first."* `y` in particular must not work."""
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert conf.CLIConfirmationGate().confirm(
        decision(), plan(), limit_price=472.62
    ) is False


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_a_closed_stdin_declines(monkeypatch, error) -> None:
    """**The inversion bug this gate was rewritten to prevent.**

    A cron, a pipe or a detached container must produce a refusal rather than
    proceeding unattended.
    """
    def raise_it(_):
        raise error()

    monkeypatch.setattr("builtins.input", raise_it)
    assert conf.CLIConfirmationGate().confirm(
        decision(), plan(), limit_price=472.62
    ) is False


def test_the_gate_has_no_parameter_that_disables_it() -> None:
    """There is deliberately no flag. `config.load_config` refuses a config
    that sets ``require_confirmation: false``, ``models.intent.Execution``
    refuses to construct one, and this is the third lock."""
    import inspect

    params = set(inspect.signature(conf.CLIConfirmationGate.confirm).parameters)
    for banned in ("require_confirmation", "skip", "force", "auto", "yes",
                   "assume_yes", "non_interactive"):
        assert banned not in params


def test_the_dry_run_gate_renders_then_declines(capsys) -> None:
    """The point of a dry run is to see exactly what would have been shown."""
    assert conf.RejectAllGate().confirm(
        decision(), plan(), limit_price=472.62, stop_price=434.81
    ) is False
    out = capsys.readouterr().out
    assert "ORDER PROPOSAL" in out
    assert DISSENT in " ".join(out.split())
    assert "Nothing was sent" in out


# --------------------------------------------------------------------------- #
# The book-rewrite gate is a different question with a different word
# --------------------------------------------------------------------------- #


def test_rewriting_the_book_needs_its_own_word(monkeypatch) -> None:
    """Not the ticker. It is a different action with a different consequence,
    and reusing the ticker would let one muscle-memory answer do both."""
    monkeypatch.setattr("builtins.input", lambda _: "TSM")
    assert conf.confirm_rewrite("config/portfolio.yaml") is False

    monkeypatch.setattr("builtins.input", lambda _: "rewrite")
    assert conf.confirm_rewrite("config/portfolio.yaml") is True


def test_a_closed_stdin_leaves_the_book_alone(monkeypatch) -> None:
    def raise_it(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_it)
    assert conf.confirm_rewrite("config/portfolio.yaml") is False


# --------------------------------------------------------------------------- #
# Preflight parsing
# --------------------------------------------------------------------------- #


def test_the_unset_commission_sentinel_becomes_none() -> None:
    """IB's "unset double" is 1.7976931348623157e+308, and printing that in a
    confirmation prompt is worse than printing nothing."""
    class State:
        commission = 1.7976931348623157e308
        initMarginChange = "100"
        maintMarginChange = "50"
        equityWithLoanChange = "0"
        commissionCurrency = "USD"
        warningText = ""
        status = "PreSubmitted"

    assert conf.Preflight.from_order_state(State()).commission is None


def test_a_missing_field_does_not_raise() -> None:
    """IB omits these entirely for some contract types, and a preflight that
    raised would block an order for a reporting quirk."""
    class Sparse:
        status = "PreSubmitted"

    report = conf.Preflight.from_order_state(Sparse())
    assert report.is_empty
    assert report.init_margin_change is None
