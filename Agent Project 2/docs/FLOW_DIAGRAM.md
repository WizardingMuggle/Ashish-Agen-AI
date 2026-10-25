# Flow Diagrams

Each box has a code (`[S1]`, `[L2]`, `[P2.1]`, …). The same code appears as a header in the terminal logs, so you can match what you read here to what you see there.

---

## 1. The whole question

```mermaid
flowchart TD
    Q([You: a request]) --> S1["[S1] CONTEXT<br/>system prompt + long-term memory<br/>+ last turns + allowed tools<br/><i>agent/app.py</i>"]
    S1 --> MODE{agent_mode}
    MODE -->|react| L["[L1..Ln] ReAct loop<br/><i>core/react_agent.py</i>"]
    MODE -->|plan| P1["[P1] PLAN<br/>JSON list of steps<br/><i>core/planner.py</i>"]
    P1 --> P2["[P2.k] EXECUTE step k<br/>a fresh ReAct executor"]
    P2 -->|next step| P2
    P2 --> P3["[P3] SYNTHESIZE<br/>answer from step results"]
    MODE -->|multi| SUP["Supervisor ReAct loop<br/>tools = specialists<br/><i>core/supervisor.py</i>"]
    L --> R{use_reflection?}
    P3 --> R
    SUP --> R
    R -->|yes| R1["[R1] CRITIQUE<br/>answer vs. evidence<br/><i>core/reflection.py</i>"]
    R1 -->|revise| REV["[R1] REVISE<br/>issues sent back to the agent"]
    REV --> F1
    R1 -->|pass| F1
    R -->|no| F1["[F1] ANSWER<br/>tokens per call, cost, trace saved"]
```

---

## 2. Inside the ReAct loop (one agent run)

```mermaid
flowchart TD
    START([task + history]) --> B{budget left?}
    B -->|no| FORCE
    B -->|yes| THINK["[Ln] THINK<br/>chat(messages, tools=schemas)<br/><i>llm.py</i>"]
    THINK --> TC{reply has<br/>tool_calls?}
    TC -->|no| ANS([FINAL ANSWER<br/>the model decided it is done])
    TC -->|yes| APPEND["append the assistant message<br/>(with its tool_calls)"]
    APPEND --> EACH["[Ln] ACT, for each tool call"]
    EACH --> ALLOW{tool in this<br/>agent's allow-list?}
    ALLOW -->|no| ERR1["result = ERROR: no such tool"]
    ALLOW -->|yes| VALID{arguments valid<br/>for the schema?}
    VALID -->|no| ERR2["result = ERROR: invalid arguments"]
    VALID -->|yes| SIDE{side effect?}
    SIDE -->|no| RUN["run the tool<br/><i>tools/registry.py</i>"]
    SIDE -->|yes| ASK{"human approves?<br/><i>core/guardrails.py</i>"}
    ASK -->|yes / always| RUN
    ASK -->|no / reason| DENY["result = DENIED: reason"]
    RUN --> OBS
    ERR1 --> OBS
    ERR2 --> OBS
    DENY --> OBS["[Ln] OBSERVE<br/>append role=tool message<br/>with the result"]
    OBS --> MAX{steps left?}
    MAX -->|yes| B
    MAX -->|no| FORCE["[L!] FORCED ANSWER<br/>tool_choice = none"]
    FORCE --> ANS2([best-effort answer])
```

Errors, denials and unknown tools are **not** exceptions. All of them become a tool result that the model reads in the next step, which is how an agent corrects itself.

---

## 3. What the messages look like after two steps

This is the agent's entire working memory. It is re-sent in full on **every** call:

```
[0] system     You are the Nimbus Ops Assistant ... (rules, memory)
[1] user       Is Priya due for a laptop refresh?
[2] assistant  "I'll look up Priya and the refresh policy."
               tool_calls: find_employee({"query":"Priya"}), search_handbook({"query":"laptop refresh"})
[3] tool       (id of call 1) [{"id": "E102", "name": "Priya Natarajan", ... "laptop_purchased": "2023-08-14"}]
[4] tool       (id of call 2) [Equipment > Laptop refresh cycle] Every employee laptop is replaced after 36 months ...
[5] assistant  "I need today's date to compute the laptop's age."
               tool_calls: get_current_datetime({})
[6] tool       (id of call 3) 2026-10-25 14:02 (Sunday) ...
--> next call: THINK on messages [0..6]
```

Type `/verbosity 3` to see this table in the logs at every step.

---

## 4. Multi-agent: specialists are tools

```mermaid
flowchart LR
    U([You]) --> S[Supervisor<br/>tools: ask_researcher,<br/>ask_analyst, ask_operator]
    S -->|"ask_researcher(task)"| R[Researcher loop<br/>search_handbook, find_employee,<br/>list_team, dates]
    S -->|"ask_analyst(task)"| A[Analyst loop<br/>check_inventory,<br/>list_low_stock, calculator]
    S -->|"ask_operator(task)"| O[Operator loop<br/>tickets, workspace files]
    R -->|report| S
    A -->|report| S
    O -->|report| S
    S --> F([final answer])
```

Each specialist gets **only** its task string and starts with a fresh message list. It does not see the conversation.

---

## 5. File-to-step table

| Code | Step | File |
|---|---|---|
| S1 | Assemble context: prompt, memory, tools | `agent/app.py`, `agent/prompts.py` |
| L*n* THINK | One model call with tool schemas | `agent/core/react_agent.py`, `agent/llm.py` |
| L*n* ACT | Allow-list, validation, approval, execution | `agent/core/react_agent.py`, `agent/tools/registry.py`, `agent/core/guardrails.py` |
| L*n* OBSERVE | Append the tool result | `agent/core/react_agent.py` |
| L! | Forced final answer (step limit or budget) | `agent/core/react_agent.py` |
| P1 / P2.k / P3 | Plan, execute steps, synthesize | `agent/core/planner.py` |
| (multi) | Supervisor + specialists as tools | `agent/core/supervisor.py` |
| R*n* | Critique / revise | `agent/core/reflection.py` |
| F1 | Answer, cost per call, trace | `agent/app.py`, `agent/usage.py`, `agent/core/trace.py` |
