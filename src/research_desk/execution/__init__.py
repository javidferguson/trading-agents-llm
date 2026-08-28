"""The ``execute`` process. **The only package that may import ib_async.**

Enforced by ``tests/test_layering.py``. The rule is one half of the §0 split:

    [ decide ]  agents + LLM + free data  ->  proposal.json  ->  [ execute ]
     no IB connection, no ib_async import                  human gate + ib_async

Nothing is imported eagerly here. ``safety`` and (from Stage 7) ``broker`` and
``confirmation`` pull ``ib_async`` at module scope, and ``decide`` legitimately
wants ``execution.journal`` without dragging the broker stack in behind it. So
import the submodule you need, explicitly::

    from research_desk.execution.journal import Journal   # no ib_async
    from research_desk.execution.safety import assert_paper_account  # ib_async

Still to arrive, at Stage 7: ``confirmation.py`` (carried, renderer rewritten for
``FinalDecision`` and gaining ``dissent``, ``invalidation`` and an ``expires_at``
refusal) and ``broker.py``.
"""
