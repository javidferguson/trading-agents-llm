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

As built at Stage 7:

* ``confirmation.py`` -- carried from the ORB engine, renderer rewritten for
  ``FinalDecision`` + ``OrderPlan``, and it now shows the ``dissent`` and the
  ``invalidation`` §4 marks required *"because both surface in the confirmation
  prompt"*. An expired proposal raises before a prompt is drawn.
* ``broker.py`` -- the only module that calls ``placeOrder``. Marketable limit,
  never market; ``assert_paper_account`` immediately before the order; a
  protective stop attached to every BUY.
* ``reconcile.py`` -- book versus account. A mismatch refuses rather than
  warning, because the share count came from the book.
* ``cli.py`` -- the ``execute`` entry point. Never scheduled (§15.8).
"""
