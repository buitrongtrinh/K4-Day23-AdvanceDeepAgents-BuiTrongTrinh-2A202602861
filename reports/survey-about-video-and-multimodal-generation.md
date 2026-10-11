# A Survey of Video and Multimodal Generation

## TL;DR
- Video generation has evolved from simple synthesis to complex "world models" that simulate physical dynamics, achieving high temporal coherence and semantic alignment [1][2].
- Modern architectures increasingly rely on hybrid approaches, where Multimodal Large Language Models (MLLMs) act as semantic planners for Diffusion Transformer (DiT) models [3][4].
- Evaluation methods are shifting from static metrics to agent-based and MLLM-driven benchmarks that prioritize human alignment and intrinsic faithfulness [5][6].
- Major open challenges persist, including error accumulation over long durations, robust physical reasoning, and computational efficiency [7][8][9].

## Background
Generative AI has experienced a paradigm shift in the last two years, moving beyond static image synthesis to dynamic, temporally grounded video and multimodal generation [1][10]. This evolution is driven by the demand for high-fidelity simulation in fields such as robotics, medical training, and content creation [11][12]. Foundational models, once limited to GANs and simple CNNs, have been superseded by latent diffusion models and Transformer-based architectures, which provide the scalability required for complex, multi-objective generation tasks [1][13]. 

Understanding these developments is critical as these systems increasingly serve as "world models" for the physical world, promising applications in simulation, robotic policy learning, and even clinical reasoning in medical AI [11][10][14]. As we move further into 2026, the convergence of generative AI with multimodal foundation models has transformed the landscape, enabling new innovations in text-to-media generation and paving the way for more controllable, efficient, and semantically grounded AI systems [10][15]. The shift toward these integrated, large-scale architectures marks a departure from traditional narrow AI, as models now begin to internalize physical laws and complex causal relationships, albeit with substantial room for improvement in accuracy and reliability [1][9].

## Diffusion-Based Video Generation
Video generation has become dominated by Diffusion Transformers (DiTs), which excel at modeling high-resolution, temporally coherent content [13][16]. Latent video diffusion models, such as Stable Video Diffusion, have established a framework for scaling training data and achieving high fidelity, serving as a crucial step in modern video generation systems [13]. However, the computational cost associated with iterative denoising and 3D token processing remains a significant bottleneck for mainstream deployment [8]. Recent advancements like SmoothCache and parameter-efficient post-tuning methods, such as ExVideo, are addressing these efficiency challenges by optimizing inference speed and reducing resource requirements without compromising generation quality [17][18]. 

Furthermore, autoregressive (AR) video diffusion models have emerged as a promising alternative for causal streaming generation. These models incorporate techniques like "memory forcing" to manage temporal information efficiently without expanding KV cache requirements, effectively solving the "mid-horizon forgetting" problem in streaming generation [2][19]. Causal video diffusion models can now also be trained directly from image-model initializations, which removes the need for large bidirectional teachers or complex distillation pipelines typically used to improve autoregressive quality [19]. 

The industry is also moving toward unified audio-visual architectures, such as Meta's MovieGen or Google's Veo-3, which are designed to synthesize temporally grounded signals synchronously, marking a significant step toward models that can generate comprehensive media experiences rather than just silent video sequences [20]. These advancements represent a broader effort to unify spatial, temporal, and sensory dimensions, pushing video generation beyond visual synthesis into a more holistic paradigm that emphasizes the importance of cross-modal synchronization for immersion and realism, which is critical for the future of interactive and generative media.

## LLM-Based Multimodal Integration
A core trend in contemporary research is the integration of MLLMs with generative backbones to improve semantic understanding, reasoning, and controllability [3][4]. Unlike earlier approaches that relied on simple text concatenation, modern systems utilize sophisticated fusion strategies, such as explicit visual-token conditioning through multi-layer cross-attention [4]. Models like LoomVideo demonstrate the effectiveness of using a large MLLM to facilitate complex video generation and editing with substantial inference acceleration [3].
 

This hybrid approach enables a more robust interface between semantic instruction and visual output, significantly enhancing the model's ability to interpret complex user intent across diverse multimodal contexts [4][21]. Research also explores how MLLM-DiT integration can benefit from discrete semantic tokens (EMA-based), which provide a stable, expressive interface between the language encoder and the diffusion model [4]. Moreover, these models are increasingly evaluated not just on text-to-video generation but on their ability to handle diverse inputs, such as audio and depth maps, as seen in architectures like Google's Veo-3 [20][10][15].

## Benchmarking and Evaluation
As generative models grow in complexity, traditional metrics (such as FID or FVD) have become insufficient for assessing intrinsic faithfulness, human alignment, and temporal consistency [5]. The field is increasingly adopting agent-based and MLLM-driven evaluation frameworks that can analyze semantic accuracy, physical consistency, and controllability in ways that better align with human preferences [6][22]. 

Benchmarks like VBench-2.0 have introduced multi-dimensional assessment criteria, evaluating aspects such as human fidelity, controllability, creativity, physics, and commonsense [5].
 Similarly, task-specific benchmarks like UI2V-Bench and ChronoMagic-Bench are providing targeted evaluation for image-to-video synthesis and time-lapse metamorphic fidelity, respectively [23][24]. These efforts reflect a broader shift toward alignment-centered evaluation that mirrors human perception and reliability. For instance, Video-Bench utilizes MLLMs and advanced techniques to achieve systematic evaluation that correlates more strongly with human judgment than previous automated metrics [6]. Similarly, VideoGen-Eval uses an agent-based system that leverages MLLM technologies to enhance assessment accuracy and human alignment in diverse video synthesis tasks [22].

## Trends and open problems
The field is currently grappling with the limitations of "visual-only" intelligence, as highlighted by benchmarks like Physics-IQ, which show that models often achieve impressive realism without capturing underlying physical principles [9]. Addressing this "physics-blindness" is a critical open problem, alongside the ongoing struggle with error accumulation in long-form generation [7][9]. 

Furthermore, while multimodal integration has improved, managing the trade-offs between computational efficiency and temporal coherence remains a core challenge [8][12]. Future research is expected to continue exploring model-agnostic learning and zero-overhead conditioning methods to ensure these generative foundations can adapt to new evidence and complex multimodal tasks without requiring extensive retraining [3][14]. Another emerging issue is the contamination of generative datasets with incidental noise—such as personal conversations or irrelevant environmental data—which can disrupt clinical reasoning or semantic fidelity in sensitive applications like medical AI [25]. Solving these robustness and reliability problems, particularly for safety-critical tasks, will be essential for the next generation of video and multimodal models.

## References
[1] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (n.d.)
[2] Memory Forcing: Attendable Mid-Horizon History for Streaming Video Generation. arxiv. https://arxiv.org/abs/2610.11756 (2026-10-08)
[3] LoomVideo: Unifying Multimodal Inputs into Video Generation and Editing. web. https://arxiv.org/abs/2606.06042 (2026-06-04)
[4] Beyond Text Conditioning: A Systematic Study of MLLM-DiT Fusion for Video Generation. web. https://arxiv.org/html/2608.14043 (2026-08-14)
[5] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. hf-search. https://huggingface.co/papers/2503.21755 (2025-03-27)
[6] Video-Bench: Human-Aligned Video Generation Benchmark. hf-search. https://huggingface.co/papers/2504.04907 (2025-04-07)
[7] Video Generation Models: A Survey of Post-Training and Alignment. arxiv. https://arxiv.org/abs/2610.00812 (2026-09-30)
[8] Efficient Video Diffusion Models: Advancements and Challenges. arxiv. https://arxiv.org/abs/2604.15911 (2026-04-17)
[9] Do generative video models learn physical principles from watching videos?. hf-search. https://huggingface.co/papers/2501.09038 (2025-01-14)
[10] Next-Gen AIGC: a review of multimodal foundation models for text-to-media innovations. web. https://link.springer.com/article/10.1007/s11704-025-51171-9 (2026-02-20)
[11] Surgical Video Generation From Diffusion to World Models: A Survey. arxiv. https://arxiv.org/abs/2608.26214 (2026-08-26)
[12] Towards human-centered and efficient video synthesis: a survey of multimodal diffusion models. web. https://link.springer.com/article/10.1007/s10462-026-11699-z (2026-09-13)
[13] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. hf-search. https://huggingface.co/papers/2311.15127 (2023-11-25)
[14] Frozen Models, Evolving Expertise: Model-Agnostic Learning from Deployment Experience for Multimodal Medical AI. hf-daily. https://huggingface.co/papers/2610.09146 (2026-10-06)
[15] Generative AI for multimodal content: a survey with empirical and experimental evaluations. web. https://link.springer.com/article/10.1007/s10462-026-11525-6 (2026-03-19)
[16] Open-Sora: Democratizing Efficient Video Production for All. hf-search. https://huggingface.co/papers/2412.20404 (2024-12-29)
[17] ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning. hf-search. https://huggingface.co/papers/2406.14130 (2024-06-20)
[18] SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers. hf-search. https://huggingface.co/papers/2411.10510 (2024-11-15)
[19] Conditional Residual Prediction: Improving Autoregressive Video Diffusion without a Bidirectional Teacher. arxiv. https://arxiv.org/abs/2610.11479 (2026-10-08)
[20] Audio-Visual Intelligence in Large Foundation Models. arxiv. https://arxiv.org/abs/2605.04045 (2026-05-05)
[21] Controllable Video Generation: A Survey. web. https://arxiv.science/abs/2507.16869 (2026-01-19)
[22] VideoGen-Eval: Agent-based System for Video Generation Evaluation. hf-search. https://huggingface.co/papers/2503.23452 (2025-04-27)
[23] ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation. hf-search. https://huggingface.co/papers/2406.18522 (2024-06-26)
[24] UI2V-Bench: An Understanding-based Image-to-video Generation Benchmark. hf-search. https://huggingface.co/papers/2509.24427 (2025-09-29)
[25] Incidental information contaminates patient notes and disrupts clinical reasoning in large language models. hf-daily. https://huggingface.co/papers/2610.08585 (2026-10-06)
