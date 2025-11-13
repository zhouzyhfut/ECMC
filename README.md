# Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding *(AAAI 2026)*  
---

## 🌟 Overview  
Understanding mental health disorders requires integrating both **emotional** and **cognitive** factors. However, existing multi-modal approaches often frame the task as simple classification, leading to limited interpretability—especially regarding the underlying emotional and cognitive mechanisms.  

To address these challenges, we propose **ECMC (Emotion–Cognition Multi-modal Captioning)**, a novel framework designed to generate **natural language descriptions** of emotional and cognitive states from **audio, visual, and textual** inputs. ECMC employs an **encoder–decoder architecture**, where modality-specific encoders are connected through a **dual-stream BridgeNet** based on Q-former. This design enables fine-grained emotion–cognition feature fusion, further enhanced via **contrastive learning**. A **LLaMA-based decoder** then aligns these features with annotated captions to produce interpretable emotion–cognition descriptions and profiles.  

Extensive objective and subjective evaluations demonstrate that ECMC not only **outperforms existing multi-modal LLMs and mental health models** in caption generation, but also **enhances diagnostic interpretability** by bridging perceptual cues and psychological reasoning.

