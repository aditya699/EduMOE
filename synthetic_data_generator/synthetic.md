Synthetic Data Generation Using Large Language Models: Advances in Text and Code

Authors: Mihai Nădaş, Laura Dioşan, Andreea Tomescu
Institution(s): Babeş-Bolyai University, KlusAI Research Lab
Date: July 23, 2025
arXiv: arXiv:2503.14023v2

📌 Overview

This paper surveys how Large Language Models (LLMs) are transforming synthetic training data generation in both natural language and programming code domains. By producing artificial yet task-relevant data, LLMs can augment or even substitute scarce, expensive, or sensitive real-world datasets.

The survey organizes recent techniques, applications, challenges, and open directions, positioning LLM-driven synthetic data as a cornerstone for the future of AI research and development.

🧠 Key Contributions

Unified Framework: Cross-domain taxonomy for synthetic data generation (text + code).

Techniques Reviewed:

Prompt-based augmentation (zero-/few-shot, topic-controlled).

Retrieval-augmented pipelines.

Iterative self-instruct and self-refinement methods.

Reinforcement learning with feedback (esp. execution feedback for code).

Applications:

Text: classification, QA, instruction tuning, reasoning datasets.

Code: code synthesis, translation, refactoring, bug repair, and coding QA.

Challenges Addressed: Factuality, realism, bias amplification, model collapse, cost trade-offs.

Future Research Directions: Automated prompt engineering, multimodal data synthesis, active learning, evaluation frameworks, and ethics.

📊 Highlights

Synthetic augmentation yields 3–26% improvements in low-data text classification scenarios.

Code-specific pipelines leverage execution feedback to validate correctness.

Prominent datasets:

Text: WANLI, GPT3Mix, Unnatural Instructions, Self-Instruct (Alpaca).

Code: Code Alpaca, WizardCoder, Magicoder, AlphaCode, Reflexion.

Model collapse risk identified when recursively training on synthetic-only data; mitigated by mixing real + synthetic.

⚠️ Challenges & Risks

Factual inaccuracies (text hallucinations).

Bias amplification from LLMs into synthetic corpora.

Distribution shift: synthetic data being “too clean” vs. messy real-world data.

Ethical/legal issues: privacy leaks, copyrighted data regeneration.

🔮 Future Directions

Standardized taxonomies and pipelines for reproducibility.

Automated, optimized prompt engineering.

Cross-modal data generation (text–image, text–audio).

Human-in-the-loop pipelines for balancing quality vs. scale.

Robust evaluation with statistical testing and real-data anchors.