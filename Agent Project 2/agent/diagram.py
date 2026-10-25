"""The flow diagram printed at start-up and by the /diagram command."""

FLOW_DIAGRAM = r"""
 ================================= EVERY QUESTION =================================

   You: "Is Priya due for a new laptop? If so, open a ticket."
            |
            v
   [S1] CONTEXT: system prompt + long-term memory + last turns + allowed tools
            |
            +-------------------- agent_mode ? ----------------------+
            |                          |                             |
          react                       plan                         multi
            |                          |                             |
            v                          v                             v
   +---- [L] REACT LOOP ----+   [P1] PLAN (JSON steps)        SUPERVISOR = REACT LOOP
   |                        |          |                      whose tools are agents:
   | THINK  LLM + schemas   |   [P2.k] for each step:          ask_researcher --+
   |   |                    |          a ReAct executor        ask_analyst    --+-> each runs
   |   +- tool_calls? ------+          runs it                 ask_operator   --+   its OWN loop
   |   |    yes             |          |                                           (indented)
   |   v                    |   [P3] SYNTHESIZE answer
   | ACT  allow-list        |
   |      approval? (y/n)   |   guardrails in every loop:
   |      run the tool      |     max_iterations -> forced answer
   |   |                    |     max_cost_usd   -> forced answer
   | OBSERVE result ->      |     approval       -> "DENIED" result
   |   tool message --------+     allow-list     -> "no such tool" result
   |                        |
   |   no tool_calls        |
   |   = FINAL ANSWER       |
   +------------------------+
            |
            v
   [R1] REFLECTION (optional): critic checks answer vs. evidence -> pass / revise
            |
            v
   [F1] ANSWER + tokens per call + cost + trace saved to traces/
"""
