#### 1.FusionSF: Fuse Heterogeneous Modalities in a Vector Quantized Framework for Robust Solar Power Forecasting. [Link](http://arxiv.org/abs/2402.05823v1) 
#### GPT Score: 96
Innovation:19, Newness:19, Potential:20, Clarity:19, Relevance:19
### Summary
Accurate solar power forecasting is crucial to integrate photovoltaic plants
into the electric grid, schedule and secure the power grid safety. This problem
becomes more demanding for those newly installed solar plants which lack
sufficient data. Current research predominantly relies on historical solar
power data or numerical weather prediction in a single-modality format,
ignoring the complementary information provided in different modalities. In
this paper, we propose a multi-modality fusion framework to integrate
historical power data, numerical weather prediction, and satellite images,
significantly improving forecast performance. We introduce a vector quantized
framework that aligns modalities with varying information densities, striking a
balance between integrating sufficient information and averting model
overfitting. Our framework demonstrates strong zero-shot forecasting
capability, which is especially useful for those newly installed plants.
Moreover, we collect and release a multi-modal solar power (MMSP) dataset from
real-world plants to further promote the research of multi-modal solar
forecasting algorithms. Our extensive experiments show that our model not only
operates with robustness but also boosts accuracy in both zero-shot forecasting
and scenarios rich with training data, surpassing leading models. We have
incorporated it into our eForecaster platform and deployed it for more than 300
solar plants with a capacity of over 15GW.

#### 2.Unveiling Group-Specific Distributed Concept Drift: A Fairness Imperative in Federated Learning. [Link](http://arxiv.org/abs/2402.07586v1) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
In the evolving field of machine learning, ensuring fairness has become a
critical concern, prompting the development of algorithms designed to mitigate
discriminatory outcomes in decision-making processes. However, achieving
fairness in the presence of group-specific concept drift remains an unexplored
frontier, and our research represents pioneering efforts in this regard.
Group-specific concept drift refers to situations where one group experiences
concept drift over time while another does not, leading to a decrease in
fairness even if accuracy remains fairly stable. Within the framework of
federated learning, where clients collaboratively train models, its distributed
nature further amplifies these challenges since each client can experience
group-specific concept drift independently while still sharing the same
underlying concept, creating a complex and dynamic environment for maintaining
fairness. One of the significant contributions of our research is the
formalization and introduction of the problem of group-specific concept drift
and its distributed counterpart, shedding light on its critical importance in
the realm of fairness. In addition, leveraging insights from prior research, we
adapt an existing distributed concept drift adaptation algorithm to tackle
group-specific distributed concept drift which utilizes a multi-model approach,
a local group-specific drift detection mechanism, and continuous clustering of
models over time. The findings from our experiments highlight the importance of
addressing group-specific concept drift and its distributed counterpart to
advance fairness in machine learning.

#### 3.Advancing Data-driven Weather Forecasting: Time-Sliding Data Augmentation of ERA5. [Link](http://arxiv.org/abs/2402.08185v1) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
Modern deep learning techniques, which mimic traditional numerical weather
prediction (NWP) models and are derived from global atmospheric reanalysis
data, have caused a significant revolution within a few years. In this new
paradigm, our research introduces a novel strategy that deviates from the
common dependence on high-resolution data, which is often constrained by
computational resources, and instead utilizes low-resolution data (2.5 degrees)
for global weather prediction and climate data analysis. Our main focus is
evaluating data-driven weather prediction (DDWP) frameworks, specifically
addressing sample size adequacy, structural improvements to the model, and the
ability of climate data to represent current climatic trends. By using the
Adaptive Fourier Neural Operator (AFNO) model via FourCastNet and a proposed
time-sliding method to inflate the dataset of the ECMWF Reanalysis v5 (ERA5),
this paper improves on conventional approaches by adding more variables and a
novel approach to data augmentation and processing. Our findings reveal that
despite the lower resolution, the proposed approach demonstrates considerable
accuracy in predicting atmospheric conditions, effectively rivaling
higher-resolution models. Furthermore, the study confirms the model's
proficiency in reflecting current climate trends and its potential in
predicting future climatic events, underscoring its utility in climate change
strategies. This research marks a pivotal step in the realm of meteorological
forecasting, showcasing the feasibility of lower-resolution data in producing
reliable predictions and opening avenues for more accessible and inclusive
climate modeling. The insights gleaned from this study not only contribute to
the advancement of climate science but also lay the groundwork for future
innovations in the field.

#### 4.LLaGA: Large Language and Graph Assistant. [Link](http://arxiv.org/abs/2402.08170v1) 
#### GPT Score: 96
Innovation:18, Newness:19, Potential:20, Clarity:20, Relevance:19
### Summary
Graph Neural Networks (GNNs) have empowered the advance in graph-structured
data analysis. Recently, the rise of Large Language Models (LLMs) like GPT-4
has heralded a new era in deep learning. However, their application to graph
data poses distinct challenges due to the inherent difficulty of translating
graph structures to language. To this end, we introduce the \textbf{L}arge
\textbf{L}anguage \textbf{a}nd \textbf{G}raph \textbf{A}ssistant
(\textbf{LLaGA}), an innovative model that effectively integrates LLM
capabilities to handle the complexities of graph-structured data. LLaGA retains
the general-purpose nature of LLMs while adapting graph data into a format
compatible with LLM input. LLaGA achieves this by reorganizing graph nodes to
structure-aware sequences and then mapping these into the token embedding space
through a versatile projector. LLaGA excels in versatility, generalizability
and interpretability, allowing it to perform consistently well across different
datasets and tasks, extend its ability to unseen datasets or tasks, and provide
explanations for graphs. Our extensive experiments across popular graph
benchmarks show that LLaGA delivers outstanding performance across four
datasets and three tasks using one single model, surpassing state-of-the-art
graph models in both supervised and zero-shot scenarios. Our code is available
at \url{https://github.com/ChenRunjin/LLaGA}

#### 5.ChemLLM: A Chemical Large Language Model. [Link](http://arxiv.org/abs/2402.06852v1) 
#### GPT Score: 96
Innovation:20, Newness:20, Potential:20, Clarity:18, Relevance:18
### Summary
Large language models (LLMs) have made impressive progress in chemistry
applications, including molecular property prediction, molecular generation,
experimental protocol design, etc. However, the community lacks a
dialogue-based model specifically designed for chemistry. The challenge arises
from the fact that most chemical data and scientific knowledge are primarily
stored in structured databases, and the direct use of these structured data
compromises the model's ability to maintain coherent dialogue. To tackle this
issue, we develop a novel template-based instruction construction method that
transforms structured knowledge into plain dialogue, making it suitable for
language model training. By leveraging this approach, we develop ChemLLM, the
first large language model dedicated to chemistry, capable of performing
various tasks across chemical disciplines with smooth dialogue interaction.
ChemLLM beats GPT-3.5 on all three principal tasks in chemistry, i.e., name
conversion, molecular caption, and reaction prediction, and surpasses GPT-4 on
two of them. Remarkably, ChemLLM also shows exceptional adaptability to related
mathematical and physical tasks despite being trained mainly on
chemical-centric corpora. Furthermore, ChemLLM demonstrates proficiency in
specialized NLP tasks within chemistry, such as literature translation and
cheminformatic programming. ChemLLM opens up a new avenue for exploration
within chemical studies, while our method of integrating structured chemical
knowledge into dialogue systems sets a new frontier for developing LLMs across
various scientific fields. Codes, Datasets, and Model weights are publicly
accessible at hf.co/AI4Chem/ChemLLM-7B-Chat.

#### 6.Mapping the Ethics of Generative AI: A Comprehensive Scoping Review. [Link](http://arxiv.org/abs/2402.08323v1) 
#### GPT Score: 95
Innovation:20, Newness:20, Potential:20, Clarity:17, Relevance:18
### Summary
The advent of generative artificial intelligence and the widespread adoption
of it in society engendered intensive debates about its ethical implications
and risks. These risks often differ from those associated with traditional
discriminative machine learning. To synthesize the recent discourse and map its
normative concepts, we conducted a scoping review on the ethics of generative
artificial intelligence, including especially large language models and
text-to-image models. Our analysis provides a taxonomy of 378 normative issues
in 19 topic areas and ranks them according to their prevalence in the
literature. The study offers a comprehensive overview for scholars,
practitioners, or policymakers, condensing the ethical debates surrounding
fairness, safety, harmful content, hallucinations, privacy, interaction risks,
security, alignment, societal impacts, and others. We discuss the results,
evaluate imbalances in the literature, and explore unsubstantiated risk
scenarios.

#### 7.Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series. [Link](http://arxiv.org/abs/2402.06295v1) 
#### GPT Score: 95
Innovation:18, Newness:20, Potential:20, Clarity:18, Relevance:19
### Summary
Electronic health records (EHR) is an inherently multimodal register of the
patient's health status characterized by static data and multivariate time
series (MTS). While MTS are a valuable tool for clinical prediction, their
fusion with other data modalities can possibly result in more thorough insights
and more accurate results. Deep neural networks (DNNs) have emerged as
fundamental tools for identifying and defining underlying patterns in the
healthcare domain. However, fundamental improvements in interpretability are
needed for DNN models to be widely used in the clinical setting. In this study,
we present an approach built on a collection of interpretable multimodal
data-driven models that may anticipate and understand the emergence of
antimicrobial multidrug resistance (AMR) germs in the intensive care unit (ICU)
of the University Hospital of Fuenlabrada (Madrid, Spain). The profile and
initial health status of the patient are modeled using static variables, while
the evolution of the patient's health status during the ICU stay is modeled
using several MTS, including mechanical ventilation and antibiotics intake. The
multimodal DNNs models proposed in this paper include interpretable principles
in addition to being effective at predicting AMR and providing an explainable
prediction support system for AMR in the ICU. Furthermore, our proposed
methodology based on multimodal models and interpretability schemes can be
leveraged in additional clinical problems dealing with EHR data, broadening the
impact and applicability of our results.

#### 8.Unmasking honey adulteration : a breakthrough in quality assurance through cutting-edge convolutional neural network analysis of thermal images. [Link](http://arxiv.org/abs/2402.08122v1) 
#### GPT Score: 95
Innovation:20, Newness:19, Potential:19, Clarity:19, Relevance:18
### Summary
Honey, a natural product generated from organic sources, is widely recognized
for its revered reputation. Nevertheless, honey is susceptible to adulteration,
a situation that has substantial consequences for both the well-being of the
general population and the financial well-being of a country. Conventional
approaches for detecting honey adulteration are often associated with extensive
time requirements and restricted sensitivity. This paper presents a novel
approach to address the aforementioned issue by employing Convolutional Neural
Networks (CNNs) for the classification of honey samples based on thermal
images. The use of thermal imaging technique offers a significant advantage in
detecting adulterants, as it can reveal differences in temperature in honey
samples caused by variations in sugar composition, moisture levels, and other
substances used for adulteration. To establish a meticulous approach to
categorizing honey, a thorough dataset comprising thermal images of authentic
and tainted honey samples was collected. Several state-of-the-art Convolutional
Neural Network (CNN) models were trained and optimized using the dataset that
was gathered. Within this set of models, there exist pre-trained models such as
InceptionV3, Xception, VGG19, and ResNet that have exhibited exceptional
performance, achieving classification accuracies ranging from 88% to 98%.
Furthermore, we have implemented a more streamlined and less complex
convolutional neural network (CNN) model, outperforming comparable models with
an outstanding accuracy rate of 99%. This simplification offers not only the
sole advantage of the model, but it also concurrently offers a more efficient
solution in terms of resources and time. This approach offers a viable way to
implement quality control measures in the honey business, so guaranteeing the
genuineness and safety of this valuable organic commodity.

#### 9.Conditional Neural Expert Processes for Learning from Demonstration. [Link](http://arxiv.org/abs/2402.08424v1) 
#### GPT Score: 95
Innovation:20, Newness:19, Potential:19, Clarity:19, Relevance:18
### Summary
Learning from Demonstration (LfD) is a widely used technique for skill
acquisition in robotics. However, demonstrations of the same skill may exhibit
significant variances, or learning systems may attempt to acquire different
means of the same skill simultaneously, making it challenging to encode these
motions into movement primitives. To address these challenges, we propose an
LfD framework, namely the Conditional Neural Expert Processes (CNEP), that
learns to assign demonstrations from different modes to distinct expert
networks utilizing the inherent information within the latent space to match
experts with the encoded representations. CNEP does not require supervision on
which mode the trajectories belong to. Provided experiments on artificially
generated datasets demonstrate the efficacy of CNEP. Furthermore, we compare
the performance of CNEP with another LfD framework, namely Conditional Neural
Movement Primitives (CNMP), on a range of tasks, including experiments on a
real robot. The results reveal enhanced modeling performance for movement
primitives, leading to the synthesis of trajectories that more accurately
reflect those demonstrated by experts, particularly when the model inputs
include intersection points from various trajectories. Additionally, CNEP
offers improved interpretability and faster convergence by promoting expert
specialization. Furthermore, we show that the CNEP model accomplishes obstacle
avoidance tasks with a real manipulator when provided with novel start and
destination points, in contrast to the CNMP model, which leads to collisions
with the obstacle.

#### 10.Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey. [Link](http://arxiv.org/abs/2402.02242v2) 
#### GPT Score: 95
Innovation:19, Newness:18, Potential:20, Clarity:19, Relevance:19
### Summary
Large-scale pre-trained vision models (PVMs) have shown great potential for
adaptability across various downstream vision tasks. However, with
state-of-the-art PVMs growing to billions or even trillions of parameters, the
standard full fine-tuning paradigm is becoming unsustainable due to high
computational and storage demands. In response, researchers are exploring
parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance
of full fine-tuning with minimal parameter modifications. This survey provides
a comprehensive overview and future directions for visual PEFT, offering a
systematic review of the latest advancements. First, we provide a formal
definition of PEFT and discuss model pre-training methods. We then categorize
existing methods into three categories: addition-based, partial-based, and
unified-based. Finally, we introduce the commonly used datasets and
applications and suggest potential future research challenges. A comprehensive
collection of resources is available at
https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning.

