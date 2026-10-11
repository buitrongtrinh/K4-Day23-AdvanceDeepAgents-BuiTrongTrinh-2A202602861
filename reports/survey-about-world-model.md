# World Models in Artificial Intelligence: A Survey

## TL;DR
*   World models enable AI to learn internal representations of environment dynamics, allowing for simulation and planning beyond simple pattern matching [1][2][3].
*   Recent progress has shifted from basic latent state modeling to high-fidelity, physics-faithful video generation and embodied agent simulations [4][5][6].
*   Integrating world models into Large Language Models (LLMs) and Multimodal LLMs aims to enhance spatial reasoning, physical grounding, and longitudinal task execution [7][8][9].
*   New evaluation frameworks, such as WorldTest and WorldScore, emphasize closed-loop performance and task-success metrics over static next-frame prediction [10][11][12].

## Background
World models in artificial intelligence refer to systems designed to learn an internal representation of the environment, which allows the AI to predict how the environment will change in response to its actions [2][3]. This internal modeling, or "mental simulation," is foundational for intelligent behavior because it enables agents to simulate potential future outcomes—often described as "hallucinating" trajectories—without needing to perform expensive or dangerous real-world trials [3][13]. As early as 2018, Ha and Schmidhuber popularized the concept by demonstrating that recurrent models could learn compressed representations of game environments, setting the stage for more complex applications [2]. Today, as we push toward AGI, the importance of world models has grown, serving as the bridge between perception and decision-making in autonomous driving, robotics, and embodied agentic systems [14][15][16].

## The Evolution of World Modeling
The trajectory of world modeling in AI has moved from simple, low-dimensional recurrent models to complex, foundation-scale architectures [2][14]. Initially, researchers focused on learning compact, compressed latent state representations, which were sufficient for simple control tasks [2]. However, these methods struggled with the high-dimensional, noisy data characteristic of real-world environments.

Modern approaches have largely abandoned simple RNN-based latent dynamics in favor of more robust state-space models and latent diffusion techniques [17][18]. These new architectures are better at managing long-term dependencies and capturing the nuances of complex physical dynamics. For instance, the use of diffusion-based policy optimization allows models to explore vast action spaces more efficiently than previous methods [17]. Furthermore, the shift toward 3D and 4D world modeling signifies a move toward more structural representations of physical reality, enabling AI to reason about space, time, and causality with significantly higher fidelity [19][6].

## Vision, Video, and Embodied Systems
Video generation has become a critical testbed for contemporary world models, moving the field past open-loop prediction towards goal-directed, closed-loop behavior [20][21]. In open-loop prediction, models attempt to predict future frames blindly, often leading to blurry, physically inconsistent outcomes. In contrast, modern closed-loop world models incorporate action-conditional dynamics, where the model's internal simulation directly dictates the agent's actions, and the agent's actions update the model's environment view [20][21][6].

A major breakthrough in this domain is the integration of physics-aware guidance, such as 4D occupancy data, which helps maintain the structural integrity of scenes over long periods [6]. Similarly, work like IntactWorld introduces the concept of keeping features "intact" (uncompressed) to preserve the intricate details required for complex manipulation tasks [4]. In the field of robotics, researchers have made substantial progress with cross-embodiment models [22]. By training on large datasets from multiple types of robots, these models learn universal world dynamics that generalize better across different physical hardware, suggesting a clear path toward general-purpose embodied agents that can adapt to new physical constraints [22].

## World Models in Language and Multimodal Reasoning
While traditional LLMs are masters of linguistic patterns, they often lack true physical and spatial grounding, a critical limitation for agentic tasks [7][8]. Integrating world modeling into Multimodal Large Language Models (MLLMs) is an active area of research designed to rectify these shortcomings [7]. By grounding reasoning in latent world dynamics, MLLMs can perform more sophisticated self-planning, allowing them to break down long-horizon tasks into manageable steps [23][24][25].

Agentic frameworks are at the forefront of this shift. For example, systems like AgentGarten create code-based worlds where agents can practice, test, and evolve their strategies through interaction rather than static data reading [23]. Similarly, models like SpatialThinker demonstrate the value of explicit 3D spatial reasoning; by introducing spatial rewards, these models learn to better understand and interact with the physical layout of their environment [8]. This combination of linguistic reasoning and world-model-based grounding represents a significant step toward agents that can operate successfully in unpredictable, real-world conditions [9].

## Evaluation, Benchmarks, and Open Challenges
As world models become more complex, the methods used to evaluate them are undergoing a fundamental transformation. For years, the field relied heavily on pixel-level metrics like PSNR or SSIM for next-frame prediction [10]. However, these metrics have proven to be poor indicators of how well a model actually understands the underlying world dynamics [12].

The current trend is toward behavior-based evaluation frameworks, such as WorldTest, World-in-World, and WorldScore [10][11][12]. These benchmarks move away from static metrics, instead evaluating a model on its ability to achieve goals in interactive, closed-loop environments [10]. This transition is essential for building trust in AI, as it directly tests task success in simulated scenarios that mimic the unpredictability of the real world [11]. Despite these improvements, the field faces significant hurdles. Object-interaction fidelity remains low, camera controllability in generated video is still difficult to master, and researchers continue to struggle with the "causal gap"—the difficulty in determining if a model has truly grasped causal principles or if it is just effectively statistically interpolating its training data [12][26][27].

## Trends and open problems
In the last two years, world modeling has evolved from a niche research area into a cornerstone of AI, moving from simple 2D simulations to foundation-scale multimodal reasoning models [14][15]. We have seen a surge in "agentic" world modeling, where the model is no longer just a passive observer but an active participant that plans, tests, and learns through continual interaction [23][24].

However, significant challenges persist. One major concern is safety; as these models are used for real-world planning, their hallucinations, while useful for imagination, can pose serious cognitive and security risks when they don't align with reality [28]. Another open question is how we can effectively bridge the "sim-to-real" gap, where models trained in high-fidelity simulated worlds still struggle with the messy, stochastic nature of the real world [27]. Finally, as we scale these models, we must address the fundamental question of how large these models need to be, and whether they can ever achieve true causal understanding through self-supervised learning alone [25][26].

## References
[1] What Does it Mean for a Neural Network to Learn a “World Model”?. web. https://arxiv.org/html/2507.21513 (2025-07-29)
[2] World Models. arxiv. https://arxiv.org/abs/1803.10122 (n.d.)
[3] Recurrent World Models Facilitate Policy Evolution. web. https://proceedings.nips.cc/paper_files/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf (n.d.)
[4] IntactWorld: Joint World Modeling with Intact Features. arxiv. https://arxiv.org/abs/2610.11174 (2026-10-08)
[5] PhyWorld: Physics-Faithful World Model for Video Generation. hf-search. https://huggingface.co/papers/2605.19242 (2026-05-19)
[6] GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation. hf-search. https://huggingface.co/papers/2512.12751 (2025-12-14)
[7] WOVEN: Weaving Visual World Modeling into Multimodal LLMs. arxiv. https://arxiv.org/abs/2610.12417 (2026-10-08)
[8] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards. hf-search. https://huggingface.co/papers/2511.07403 (2025-11-10)
[9] From Images to Tasks: Characterizing Multimodal LLM Interactions in the Wild. arxiv. https://arxiv.org/abs/2610.00701 (2026-09-30)
[10] Benchmarking World-Model Learning (WorldTest / AutumnBench). web. https://arxiv.org/html/2510.19788v2 (2025-10-30)
[11] World-in-World: World Models in a Closed-Loop World. web. https://arxiv.org/html/2510.18135 (2025-10-29)
[12] WorldScore: A Unified Evaluation Benchmark for World Generation. web. https://openaccess.thecvf.com/content/ICCV2025/papers/Duan_WorldScore_A_Unified_Evaluation_Benchmark_for_World_Generation_ICCV_2025_paper.pdf (2025-06-01)
[13] World Model for Robot Learning: A Comprehensive Survey. web. https://arxiv.org/html/2605.00080 (n.d.)
[14] EVA: An Embodied World Model for Future Video Anticipation. hf-search. https://huggingface.co/papers/2410.15461 (2024-10-20)
[15] Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2). hf-search. https://huggingface.co/papers/2402.16720 (2024-02-26)
[16] Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots. hf-search. https://huggingface.co/papers/2504.16680 (2026-01-08)
[17] Scaling World-Model Reinforcement Learning Through Diffusion Policy Optimization. hf-search. https://huggingface.co/papers/2605.26282 (2026-05-25)
[18] Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient. hf-search. https://huggingface.co/papers/2410.08893 (2024-10-11)
[19] 3D and 4D World Modeling: A Survey. arxiv. https://arxiv.org/abs/2509.07996 (2025-09-04)
[20] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. arxiv. https://arxiv.org/abs/2610.12459 (2026-10-08)
[21] Video Prediction Policy 2: Predict Better, Act Better. arxiv. https://arxiv.org/abs/2610.10270 (2026-10-07)
[22] Cross-Embodiment Robot Foundation World Models with Latent Actions. arxiv. https://arxiv.org/abs/2610.10846 (2026-10-07)
[23] AgentGarten: Code Worlds for Evolving Agents. arxiv. https://arxiv.org/abs/2610.12374 (2026-10-08)
[24] EvolvingAgent: Curriculum Self-evolving Agent with Continual World Model for Long-Horizon Tasks. hf-search. https://huggingface.co/papers/2502.05907 (2026-04-29)
[25] Agentic Cognitive Depth: Operational Criteria for Evaluating LLM Agents. arxiv. https://arxiv.org/abs/2610.04168 (2026-10-03)
[26] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation. web. https://aclanthology.org/2025.findings-acl.1342/ (2025-07-01)
[27] What 30,000 Hours of Ego-centric Video Does Not Teach. arxiv. https://arxiv.org/abs/2610.12464 (2026-10-08)
[28] Safety, Security, and Cognitive Risks in World Models. arxiv. https://arxiv.org/abs/2604.01346 (2026-04-01)
