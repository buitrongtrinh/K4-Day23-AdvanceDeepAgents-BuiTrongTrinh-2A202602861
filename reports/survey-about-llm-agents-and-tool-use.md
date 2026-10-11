# LLM Agents and Tool Use: A Survey of Architectures, Benchmarks, and Security

## TL;DR
- LLM agents extend model capabilities by interleaving reasoning (e.g., Chain-of-Thought) with external tool use, transforming static LLMs into proactive problem solvers [1][2].
- The agentic lifecycle involves planning, memory, and execution control, with recent research emphasizing the "horizon gap"—the reliability challenge in complex, multi-step tasks [3].
- Evaluation is shifting toward procedural, multi-turn, and task-specific benchmarks to address the limitations of static accuracy metrics [4][5].
- Security research has expanded to address long-horizon attack vectors, including tool-chain hijacking and persistent state corruption, necessitating foresight-oriented safety frameworks [6][7].

## Background
Large Language Models (LLMs) operate primarily within the boundaries of their training data and static inference patterns. To address these limitations, LLM agents have emerged as systems capable of reasoning, planning, and executing sequences of actions using external tools [8]. Foundational paradigms like the MRKL architecture established the modular integration of LLMs with external systems [9], while subsequent work such as Toolformer demonstrated that LLMs could self-teach the use of APIs through self-supervised learning [10]. The ReAct paradigm further advanced this by showing that interleaving reasoning traces with action execution enables agents to better ground their decisions in external knowledge [1]. Today, these developments support autonomous agents across natural science, social science, and engineering domains, necessitating more rigorous evaluation and security frameworks [8].

## Paradigms of Agentic Tool Use
Agentic tool use has transitioned from basic API invocations to sophisticated architectures where LLMs act as central controllers. The current landscape is categorized by three pillars: tool use, planning, and feedback-based learning [2]. A pivotal development is the transition toward agents that don't just call tools but perform deep reasoning on the tool outputs. For instance, Toolformer demonstrated how LLMs can self-teach API usage through self-supervised learning, essentially optimizing which tools to call to minimize perplexity on future tokens [10].

The ReAct (Reasoning and Acting) paradigm, which interleaves reasoning traces with action execution, remains a foundational approach [1]. By grounding reasoning in external knowledge, this method has proven effective at reducing hallucination. Building on this, newer architectures seek to improve long-horizon reasoning. ToolTree, for instance, employs a Monte Carlo Tree Search (MCTS) approach with dual-stage evaluation and bidirectional pruning, significantly increasing the efficiency of tool-planning in environments with deep decision trees [11].

Planning frameworks are becoming increasingly integrated with reinforcement learning to handle more complex scenarios. PilotRL offers a global planning-guided training method that enhances LLM performance in long-horizon decision-making by optimizing the entire trajectory rather than individual steps [12]. Similarly, CM-DPO introduces a Constraint-Margin objective, enabling developers to encode symbolic constraints into the agent’s policy [13]. This allows for a clean separation between hard constraints (like safety or format requirements) and soft constraints, which are often the source of agent unpredictability.

Frameworks like AgentLite and PoAct represent the shift towards flexible, modular agent development. AgentLite provides a lightweight library for rapidly prototyping multi-agent systems and new reasoning strategies [14]. PoAct, in contrast, focuses on generalized applications by implementing a policy and action dual-control mechanism, which allows the model to dynamically switch between different reasoning modes depending on the task’s requirements [15]. This adaptability is crucial for agents operating in heterogeneous, unpredictable environments.

## Evaluation Frameworks and Benchmarks
As agentic capabilities have expanded, the limitations of simple accuracy-based metrics have become increasingly apparent. Traditional evaluations often fail to capture the procedural nature of an agent’s success, leading to an "illusion of success" where an agent might provide the correct answer through a flawed, or even dangerous, reasoning path [5]. To combat this, recent research advocates for procedure-aware evaluation, which considers the trajectory and the intermediate steps an agent takes [16].

A two-dimensional taxonomy that distinguishes between the objective of the evaluation (e.g., robustness, efficiency, correctness) and the process (e.g., automated, human-in-the-loop, agent-as-a-judge) is now standard [4]. Specialized benchmarks have emerged for specific domains, such as TelcoAgent-Bench for complex telecommunication operations [17] and Mobile-bench for mobile-based task completion [18]. These benchmarks are designed to mimic real-world interactions rather than rely on static datasets.

For multi-turn, iterative agents, analytical evaluation platforms provide leaderboards that assess performance across multiple levels of task complexity [19]. Safety-focused benchmarks, such as SafeAgentBench, represent another crucial frontier, specifically evaluating how well agents adhere to safety constraints during the planning and execution stages [20]. Furthermore, the scalable "agent-as-a-judge" methodology is gaining traction, using highly-capable LLMs to provide feedback and evaluation on the trajectories of smaller, specialized agents [21]. This shift towards more granular, process-aware evaluation is essential for moving agent technology from the lab into mission-critical, real-world deployment.

## Security and Trustworthiness
The integration of external tools and long-horizon planning has increased the security risks for AI systems [22]. Unlike the transient and often single-turn nature of prompt injection in LLMs, modern agentic security risks involve complex, long-horizon attacks such as intent hijacking, tool-chain manipulation, and objective drifting [6]. The "horizon gap"—the inherent difficulty in planning, executing, and evaluating tasks over long periods—further complicates security, as it is often impossible for designers to anticipate the delayed, cascading consequences of an agent’s early-stage actions [3].

Research indicates that even sophisticated agents remain highly susceptible to multi-step failure modes [6]. For instance, an agent might be tricked into using a legitimate tool in a malicious way, effectively enabling an attacker to bypass traditional sandboxing [23]. To mitigate this, recent developments like JANUS provide a foresight-oriented framework for long-horizon agent safety [7]. JANUS employs Coupled Anticipation and Adjudication Reinforcement Learning (CoAA-RL) to jointly optimize safety forecasting and adjudication, proactively identifying risks before they materialize into harmful actions.

Other approaches focus on the real-time monitoring of agent trajectories. OnTrack, for example, utilizes streaming structure-aware optimal transport to compare an agent’s current execution against known safe trajectories [24]. This allows for interventions during runtime, rather than just post-hoc assessment. Comprehensive survey frameworks for secure agent design now emphasize the need for a shift in perspective, moving away from simple input sanitization towards a systems-oriented framework that incorporates explicit trust boundaries, privilege control, and provenance-aware state management [23][25]. As agents increasingly operate autonomously with high-level access to sensitive infrastructure, these multilayered security and safety frameworks will become a fundamental requirement for reliable deployment.

## Trends and open problems
The field has moved toward unifying disparate agentic practices under common taxonomies and lifecycle frameworks [8][3]. However, key challenges persist. Reliable execution over long horizons remains elusive, with research focusing on the intersection of memory management and planning control [3]. There is also a move to bridge the gap between simulated benchmarks and real-world deployment, where reliability and compliance are paramount [4]. Future research is expected to continue focusing on aligning agents with user intent through continuous feedback learning and more robust, foresight-based safety mechanisms [2][7].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804 (2024-06-11)
[3] The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents. web. https://arxiv.org/pdf/2608.06663 (2026-08-01)
[4] Evaluation and Benchmarking of LLM Agents: A Survey. web. https://dl.acm.org/doi/10.1145/3711896.3736570 (2025-08-03)
[5] The Illusion of Success: Procedure-Aware Evaluation is Critical for LLM Agents. web. https://doi.org/10.1007/978-3-032-37682-4_29 (2026-10-06)
[6] AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks. web. https://proceedings.mlr.press/v306/jiang26as.html (2026-06-11)
[7] JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety. web. https://arxiv.org/html/2607.19913 (2026-07-22)
[8] A survey on large language model based autonomous agents. web. https://link.springer.com/article/10.1007/s11704-024-40231-1 (2024-03-22)
[9] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. web. https://arxiv.org/abs/2205.00445 (2022-05-01)
[10] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://ar5iv.labs.arxiv.org/html/2302.04761 (2023-02-09)
[11] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning. hf-search. https://huggingface.co/papers/2603.12740 (2026-03-13)
[12] PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning. hf-search. https://huggingface.co/papers/2508.00344 (2025-08-01)
[13] CM-DPO: Constraint-Margin Direct Preference Optimization for LLM Planning. arxiv. https://arxiv.org/abs/2610.09219 (2026-10-06)
[14] AgentLite: A Lightweight Library for Building and Advancing Task-Oriented LLM Agent System. hf-search. https://huggingface.co/papers/2402.15538 (2024-02-23)
[15] PoAct: Policy and Action Dual-Control Agent for Generalized Applications. hf-search. https://huggingface.co/papers/2501.07054 (2025-01-13)
[16] A Unified Framework for the Evaluation of LLM Agentic Capabilities. web. https://arxiv.org/html/2605.27898 (n.d.)
[17] TelcoAgent-Bench: A Multilingual Benchmark for Telecom AI Agents. web. https://doi.org/10.1109/PIMRC69400.2026.11711756 (2026-09-01)
[18] Mobile-bench: An evaluation benchmark for llm-based mobile agents. web. https://arxiv.org/abs/2407.00993 (2024-07-01)
[19] AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. web. https://arxiv.org/abs/2401.13158 (2024-01-24)
[20] SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents. web. https://arxiv.org/abs/2412.13178 (2024-12-19)
[21] Agent-as-a-judge: Evaluate agents with agents. web. https://arxiv.org/abs/2410.10934 (2024-10-15)
[22] A Comparative Survey of Security Risks in AI Systems: From LLMs to AI Agents and Embodied Agents. web. https://dl.acm.org/doi/abs/10.1145/3837083 (2026-08-22)
[23] Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation. arxiv. https://arxiv.org/abs/2606.10749 (2026-06-09)
[24] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[25] Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, and system security. arxiv. https://arxiv.org/abs/2605.23989 (2026-05-17)
