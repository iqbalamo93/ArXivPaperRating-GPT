#### 1.Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning. [Link](http://arxiv.org/abs/2402.06619v1) 
#### GPT Score: 100
Innovation:20, Newness:20, Potential:20, Clarity:20, Relevance:20
### Summary
Datasets are foundational to many breakthroughs in modern artificial
intelligence. Many recent achievements in the space of natural language
processing (NLP) can be attributed to the finetuning of pre-trained models on a
diverse set of tasks that enables a large language model (LLM) to respond to
instructions. Instruction fine-tuning (IFT) requires specifically constructed
and annotated datasets. However, existing datasets are almost all in the
English language. In this work, our primary goal is to bridge the language gap
by building a human-curated instruction-following dataset spanning 65
languages. We worked with fluent speakers of languages from around the world to
collect natural instances of instructions and completions. Furthermore, we
create the most extensive multilingual collection to date, comprising 513
million instances through templating and translating existing datasets across
114 languages. In total, we contribute four key resources: we develop and
open-source the Aya Annotation Platform, the Aya Dataset, the Aya Collection,
and the Aya Evaluation Suite. The Aya initiative also serves as a valuable case
study in participatory research, involving collaborators from 119 countries. We
see this as a valuable framework for future research collaborations that aim to
bridge gaps in resources.

#### 2.Accelerated Smoothing: A Scalable Approach to Randomized Smoothing. [Link](http://arxiv.org/abs/2402.07498v1) 
#### GPT Score: 96
Innovation:19, Newness:20, Potential:20, Clarity:19, Relevance:18
### Summary
Randomized smoothing has emerged as a potent certifiable defense against
adversarial attacks by employing smoothing noises from specific distributions
to ensure the robustness of a smoothed classifier. However, the utilization of
Monte Carlo sampling in this process introduces a compute-intensive element,
which constrains the practicality of randomized smoothing on a larger scale. To
address this limitation, we propose a novel approach that replaces Monte Carlo
sampling with the training of a surrogate neural network. Through extensive
experimentation in various settings, we demonstrate the efficacy of our
approach in approximating the smoothed classifier with remarkable precision.
Furthermore, we demonstrate that our approach significantly accelerates the
robust radius certification process, providing nearly $600$X improvement in
computation time, overcoming the computational bottlenecks associated with
traditional randomized smoothing.

#### 3.NCRF: Neural Contact Radiance Fields for Free-Viewpoint Rendering of Hand-Object Interaction. [Link](http://arxiv.org/abs/2402.05532v2) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:19, Clarity:18, Relevance:19
### Summary
Modeling hand-object interactions is a fundamentally challenging task in 3D
computer vision. Despite remarkable progress that has been achieved in this
field, existing methods still fail to synthesize the hand-object interaction
photo-realistically, suffering from degraded rendering quality caused by the
heavy mutual occlusions between the hand and the object, and inaccurate
hand-object pose estimation. To tackle these challenges, we present a novel
free-viewpoint rendering framework, Neural Contact Radiance Field (NCRF), to
reconstruct hand-object interactions from a sparse set of videos. In
particular, the proposed NCRF framework consists of two key components: (a) A
contact optimization field that predicts an accurate contact field from 3D
query points for achieving desirable contact between the hand and the object.
(b) A hand-object neural radiance field to learn an implicit hand-object
representation in a static canonical space, in concert with the specifically
designed hand-object motion field to produce observation-to-canonical
correspondences. We jointly learn these key components where they mutually help
and regularize each other with visual and geometric constraints, producing a
high-quality hand-object reconstruction that achieves photo-realistic novel
view synthesis. Extensive experiments on HO3D and DexYCB datasets show that our
approach outperforms the current state-of-the-art in terms of both rendering
quality and pose estimation accuracy.

#### 4.RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback. [Link](http://arxiv.org/abs/2402.03681v2) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
Reward engineering has long been a challenge in Reinforcement Learning (RL)
research, as it often requires extensive human effort and iterative processes
of trial-and-error to design effective reward functions. In this paper, we
propose RL-VLM-F, a method that automatically generates reward functions for
agents to learn new tasks, using only a text description of the task goal and
the agent's visual observations, by leveraging feedbacks from vision language
foundation models (VLMs). The key to our approach is to query these models to
give preferences over pairs of the agent's image observations based on the text
description of the task goal, and then learn a reward function from the
preference labels, rather than directly prompting these models to output a raw
reward score, which can be noisy and inconsistent. We demonstrate that RL-VLM-F
successfully produces effective rewards and policies across various domains -
including classic control, as well as manipulation of rigid, articulated, and
deformable objects - without the need for human supervision, outperforming
prior methods that use large pretrained models for reward generation under the
same assumptions.

#### 5.Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT. [Link](http://arxiv.org/abs/2402.07440v1) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
Retrieval pipelines-an integral component of many machine learning
systems-perform poorly in domains where documents are long (e.g., 10K tokens or
more) and where identifying the relevant document requires synthesizing
information across the entire text. Developing long-context retrieval encoders
suitable for these domains raises three challenges: (1) how to evaluate
long-context retrieval performance, (2) how to pretrain a base language model
to represent both short contexts (corresponding to queries) and long contexts
(corresponding to documents), and (3) how to fine-tune this model for retrieval
under the batch size limitations imposed by GPU memory constraints. To address
these challenges, we first introduce LoCoV1, a novel 12 task benchmark
constructed to measure long-context retrieval where chunking is not possible or
not effective. We next present the M2-BERT retrieval encoder, an 80M parameter
state-space encoder model built from the Monarch Mixer architecture, capable of
scaling to documents up to 32K tokens long. We describe a pretraining data
mixture which allows this encoder to process both short and long context
sequences, and a finetuning approach that adapts this base model to retrieval
with only single-sample batches. Finally, we validate the M2-BERT retrieval
encoder on LoCoV1, finding that it outperforms competitive baselines by up to
23.3 points, despite containing 5-90x fewer parameters.

#### 6.InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning. [Link](http://arxiv.org/abs/2402.06332v1) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
The math abilities of large language models can represent their abstract
reasoning ability. In this paper, we introduce and open-source our math
reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We
unify chain-of-thought reasoning, reward modeling, formal reasoning, data
augmentation, and code interpreter in a unified seq2seq format and supervise
our model to be a versatile math reasoner, verifier, prover, and augmenter.
These abilities can be used to develop the next math LLMs or self-iteration.
InternLM-Math obtains open-sourced state-of-the-art performance under the
setting of in-context learning, supervised fine-tuning, and code-assisted
reasoning in various informal and formal benchmarks including GSM8K, MATH,
Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves
30.3 on the MiniF2F test set without fine-tuning. We further explore how to use
LEAN to solve math problems and study its performance under the setting of
multi-task learning which shows the possibility of using LEAN as a unified
platform for solving and proving in math. Our models, codes, and data are
released at \url{https://github.com/InternLM/InternLM-Math}.

#### 7.Game-theoretic Counterfactual Explanation for Graph Neural Networks. [Link](http://arxiv.org/abs/2402.06030v1) 
#### GPT Score: 96
Innovation:18, Newness:20, Potential:20, Clarity:18, Relevance:20
### Summary
Graph Neural Networks (GNNs) have been a powerful tool for node
classification tasks in complex networks. However, their decision-making
processes remain a black-box to users, making it challenging to understand the
reasoning behind their predictions. Counterfactual explanations (CFE) have
shown promise in enhancing the interpretability of machine learning models.
Prior approaches to compute CFE for GNNS often are learning-based approaches
that require training additional graphs. In this paper, we propose a
semivalue-based, non-learning approach to generate CFE for node classification
tasks, eliminating the need for any additional training. Our results reveals
that computing Banzhaf values requires lower sample complexity in identifying
the counterfactual explanations compared to other popular methods such as
computing Shapley values. Our empirical evidence indicates computing Banzhaf
values can achieve up to a fourfold speed up compared to Shapley values. We
also design a thresholding method for computing Banzhaf values and show
theoretical and empirical results on its robustness in noisy environments,
making it superior to Shapley values. Furthermore, the thresholded Banzhaf
values are shown to enhance efficiency without compromising the quality (i.e.,
fidelity) in the explanations in three popular graph datasets.

#### 8.Memory-Efficient Vision Transformers: An Activation-Aware Mixed-Rank Compression Strategy. [Link](http://arxiv.org/abs/2402.06004v1) 
#### GPT Score: 96
Innovation:19, Newness:20, Potential:20, Clarity:18, Relevance:19
### Summary
As Vision Transformers (ViTs) increasingly set new benchmarks in computer
vision, their practical deployment on inference engines is often hindered by
their significant memory bandwidth and (on-chip) memory footprint requirements.
This paper addresses this memory limitation by introducing an activation-aware
model compression methodology that uses selective low-rank weight tensor
approximations of different layers to reduce the parameter count of ViTs. The
key idea is to decompose the weight tensors into a sum of two
parameter-efficient tensors while minimizing the error between the product of
the input activations with the original weight tensor and the product of the
input activations with the approximate tensor sum. This approximation is
further refined by adopting an efficient layer-wise error compensation
technique that uses the gradient of the layer's output loss. The combination of
these techniques achieves excellent results while it avoids being trapped in a
shallow local minimum early in the optimization process and strikes a good
balance between the model compression and output accuracy. Notably, the
presented method significantly reduces the parameter count of DeiT-B by 60%
with less than 1% accuracy drop on the ImageNet dataset, overcoming the usual
accuracy degradation seen in low-rank approximations. In addition to this, the
presented compression technique can compress large DeiT/ViT models to have
about the same model size as smaller DeiT/ViT variants while yielding up to
1.8% accuracy gain. These results highlight the efficacy of our approach,
presenting a viable solution for embedding ViTs in memory-constrained
environments without compromising their performance.

#### 9.RESMatch: Referring Expression Segmentation in a Semi-Supervised Manner. [Link](http://arxiv.org/abs/2402.05589v2) 
#### GPT Score: 96
Innovation:19, Newness:20, Potential:19, Clarity:19, Relevance:19
### Summary
Referring expression segmentation (RES), a task that involves localizing
specific instance-level objects based on free-form linguistic descriptions, has
emerged as a crucial frontier in human-AI interaction. It demands an intricate
understanding of both visual and textual contexts and often requires extensive
training data. This paper introduces RESMatch, the first semi-supervised
learning (SSL) approach for RES, aimed at reducing reliance on exhaustive data
annotation. Extensive validation on multiple RES datasets demonstrates that
RESMatch significantly outperforms baseline approaches, establishing a new
state-of-the-art. Although existing SSL techniques are effective in image
segmentation, we find that they fall short in RES. Facing the challenges
including the comprehension of free-form linguistic descriptions and the
variability in object attributes, RESMatch introduces a trifecta of
adaptations: revised strong perturbation, text augmentation, and adjustments
for pseudo-label quality and strong-weak supervision. This pioneering work lays
the groundwork for future research in semi-supervised learning for referring
expression segmentation.

#### 10.Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions. [Link](http://arxiv.org/abs/2402.05541v1) 
#### GPT Score: 95
Innovation:20, Newness:20, Potential:19, Clarity:18, Relevance:18
### Summary
Recent advancements in federated learning (FL) have produced models that
retain user privacy by training across multiple decentralized devices or
systems holding local data samples. However, these strategies often neglect the
inherent challenges of statistical heterogeneity and vulnerability to
adversarial attacks, which can degrade model robustness and fairness.
Personalized FL strategies offer some respite by adjusting models to fit
individual client profiles, yet they tend to neglect server-side aggregation
vulnerabilities. To address these issues, we propose Reinforcement Federated
Learning (RFL), a novel framework that leverages deep reinforcement learning to
adaptively optimize client contribution during aggregation, thereby enhancing
both model robustness against malicious clients and fairness across
participants under non-identically distributed settings. To achieve this goal,
we propose a meticulous approach involving a Deep Deterministic Policy
Gradient-based algorithm for continuous control of aggregation weights, an
innovative client selection method based on model parameter distances, and a
reward mechanism guided by validation set performance. Empirically, extensive
experiments demonstrate that, in terms of robustness, RFL outperforms the
state-of-the-art methods, while maintaining comparable levels of fairness,
offering a promising solution to build resilient and fair federated systems.

