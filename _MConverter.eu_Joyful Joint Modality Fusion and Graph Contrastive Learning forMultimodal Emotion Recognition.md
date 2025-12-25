Joyful: Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition

Dongyuan Li, Yusong Wang, Kotaro Funakoshi, and Manabu Okumura Tokyo Institute of Technology, Tokyo, Japan

\{lidy, wangyi, funakoshi, oku\}@lr.pi.titech.ac.jp

Abstract

Background/Topic The ship sank into the sea. ==> Sad & Fear

" 

\[Fear\] It’s getting quiet. I love you, Jack. 

𝑢\! 

Multimodal emotion recognition aims to rec-

$

ognize emotions for each utterance of multiple

𝑢

\[Surprise\] Don’t you do that. Don’t you 

\#

say goodbyes. Do you understand ? 

modalities, which has received increasing at-

𝑢" 

tention for its application in human-machine

\[Fear\] I am so cold. 

\(

interaction. Current graph-based methods fail

𝑢$%

\[Sad\] You’re gonna get out of here and you’re 

to simultaneously depict global contextual fea-

gonna make lots of babies and watch them grow. 

tures and local diverse uni-modal features in

$

a dialogue. Furthermore, with the number of

𝑢& 

\[Sad\] Winning that ticket was the best thing that 

ever happened to me. It took me to meet you. 

graph layers increasing, they easily fall into

𝑢" 

over-smoothing. In this paper, we propose a

\[sad\] I can’t feel my body. 

\)

method for joint modality fusion and graph

𝑢$' 

\[Fear\] You must promise me that you’l 

contrastive learning for multimodal emotion

survive, you won’t give up. 

recognition \(JOYFUL\), where multimodality fu-

sion, contrastive learning, and emotion recogni-

Figure 1: Emotions are affected by multiple uni-modal, 

tion are jointly optimized. Specifically, we first

global contextual, intra- and inter-person dependencies. 

design a new multimodal fusion mechanism

Images are from the movie “Titanic”. 

that can provide deep interaction and fusion be-

tween the global contextual and uni-modal spe-

cific features. Then, we introduce a graph con-

to be widely applied in other tasks such as question

trastive learning framework with inter-view and

answering \(Ossowski and Hu, 2023; Wang et al., 

intra-view contrastive losses to learn more dis-

2022b; Wang, 2022\), text generation \(Liang et al., 

tinguishable representations for samples with

2023; Zhang et al., 2023; Li et al., 2022a\) and bioin-different sentiments. Extensive experiments on

formatics \(Nicolson et al., 2023; You et al., 2022\). 

three benchmark datasets indicate that JOYFUL

Figure 1 shows that emotions expressed in a achieved state-of-the-art \(SOTA\) performance

dialogue are affected by three main factors: 1\) mul-

compared to all baselines. 

tiple uni-modalities \(different modalities complete

1 Introduction

each other to provide a more informative utterance

representation\); 2\) global contextual information

“Integration of information from multiple sensory

\(uA depends on the topic “The ship sank into the

3

channels is crucial for understanding tendencies

sea”, indicating fear\); and 3\) intra-person and inter-

and reactions in humans” \(Partan and Marler, 

person dependencies \(uA becomes sad affected by

6

1999\). Multimodal emotion recognition in conver-sadness in uB

\). Depending on how to model

4 &uB

5

sations \(MERC\) aims exactly to identify and track

intra-person and inter-person dependencies, current

the emotional state of each utterance from hetero-

MERC methods can be categorized into Sequence-

geneous visual, audio, and text channels. Due to its

based and Graph-based methods. The former \(Dai

potential applications in creating human-computer

et al., 2021; Mao et al., 2022; Liang et al., 2022\)

interaction systems \(Li et al., 2022b\), social media use recurrent neural networks or Transformers to

analysis \(Gupta et al., 2022; Wang et al., 2023\), 

model the temporal interaction between utterances. 

and recommendation systems \(Singh et al., 2022\), 

However, they failed to distinguish intra-speaker

MERC has received increasing attention in the nat-

and inter-speaker dependencies and easily lost uni-

ural language processing \(NLP\) community \(Poria

modal specific features by the cross-modal atten-

et al., 2019b, 2021\), which even has the potential tion mechanism \(Rajan et al., 2022\). Graph struc-16051

Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16051–16069

December 6-10, 2023 ©2023 Association for Computational Linguistics

ture \(Joshi et al., 2022; Wei et al., 2019\) solves trastive learning could provide more distinguish-these issues by using edges between nodes \(speak-

able node representations to benefit various down-

ers\) to distinguish intra-speaker and inter-speaker

stream tasks, we propose a cross-view GCL-based

dependencies. Graph Neural Networks \(GNNs\) fur-

framework to alleviate the difficulty of categorizing

ther help nodes learn common features by aggregat-

similar emotions, which helps to learn more distinc-

ing information from neighbours while maintaining

tive utterance representations by making samples

their uni-modal specific features. 

with the same sentiment cohesive and those with

Although graph-based MERC methods have

different sentiments mutually exclusive. Further-

achieved great success, there still remain problems

more, graph augmentation strategies are designed

that need to be solved: 1\) Current methods directly

to improve JOYFUL’s robustness and generalizabil-

aggregate features of multiple modalities \(Joshi

ity. 3\) We jointly optimize each part of JOYFUL in

et al., 2022\) or project modalities into a latent an end-to-end manner to ensure global optimized

space to learn representations \(Li et al., 2022e\), 

performance. The main contributions of this study

which ignores the diversity of each modality and

can be summarized as follows:

fails to capture richer semantic information from

• We propose a novel joint leaning framework

each modality. They also ignore global contex-

for MERC, where multimodality fusion, GCL, 

tual information during the feature fusion process, 

and emotion recognition are jointly optimized for

leading to poor performance. 2\) Since all graph-

global optimal performance. Our new multimodal

based methods adopt GNN \(Scarselli et al., 2009\)

fusion mechanism can obtain better representations

or Graph Convolutional Networks \(GCNs\) \(Kipf

by simultaneously depicting global contextual and

and Welling, 2017\), with the number of layers deep-local uni-modal specific features. 

ening, the phenomenon of over-smoothing starts

• To the best of our knowledge, JOYFUL is the

to appear, resulting in the representation of sim-

first method to utilize graph contrastive learning for

ilar sentiments being indistinguishable. 3\) Most

MERC, which significantly improves the model’s

methods use a two-phase pipeline \(Fu et al., 2021; 

ability to distinguish different sentiments. Multiple

Joshi et al., 2022\), where they first extract and fuse graph augmentation strategies further improve the

uni-modal features as utterance representations and

model’s stability and generalization. 

then fix them as input for graph models. However, 

• Extensive experiments conducted on three mul-

the two-phase pipeline will lead to sub-optimal per-

timodal benchmark datasets demonstrated the ef-

formance since the fused representations are fixed

fectiveness and robustness of JOYFUL. 

and cannot be further improved to benefit from the

2 Related Work

downstream supervisory signals. 

To solve the above-mentioned problems, we pro-

2.1 Multimodal Emotion Recognition

pose Joint multimodality fusion and graph con-

Depending on how to model the context of utter-

trastive learning for MERC \(JOYFUL\), where multi-

ances, existing MERC methods are categorized

modality fusion, graph contrastive learning \(GCL\), 

into three classes: Recurrent-based methods \(Ma-

and multimodal emotion recognition are jointly op-

jumder et al., 2019; Mao et al., 2022\) adopt RNN

timized in an overall objective function. 1\) We

or LSTM to model the sequential context for each

first design a new multimodal fusion mechanism

utterance. Transformers-based methods \(Ling et al., 

that can simultaneously learn and fuse a global

2022; Liang et al., 2022; Le et al., 2022\) use contextual representation and uni-modal specific

Transformers with cross-modal attention to model

representations. For the global contextual represen-

the intra- and inter-speaker dependencies. Graph-

tation, we smooth it with a proposed topic-related

based methods \(Joshi et al., 2022; Zhang et al., 

vector to maintain its consistency, where the topic-

2021; Fu et al., 2021\) can control context informa-related vector is temporally updated since the topic

tion for each utterance and provide accurate intra-

usually changes. For uni-modal specific represen-

and inter-speaker dependencies, achieving SOTA

tations, we project them into a shared subspace to

performance on many MERC benchmark datasets. 

fully explore their richer semantics without losing

alignment with other modalities. 2\) To alleviate

2.2 Multimodal Fusion Mechanism

the over-smoothing issue of deeper GNN layers, 

Learning effective fusion mechanisms is one of the

inspired by You et al. \(2020\), that showed con-core challenges in multimodal learning \(Shankar, 

16052





**\(A\) Uni-modal Extractor \(B\) Multimodal Fusion Module**

**\(C\) Graph Contrastive Learning Module**

**\(D\) Classifier**

***Video***

$

**Visual**

**Tr**

**Te**

**Inter-view Negative pairs**

\(O

Vi

𝑓\! 

**a**

**m**

**Frustrated**

p

s

**n**

**p**

**Intra-view Negative pairs *Edge Perturbation***

en

u

$

a

𝑧

**s**

**or**

-

l

\! 

**f**

**a**

**Positive pairs**

Fa

E

𝑥

**o**

\! 

**l**

n

**r**

**S**

c

$

**Audio**

**me**

c

e

𝑓

**m**



o

2

" 

**r**

**o**

**Angry**

d

. 

**o**

𝑢

0

e

$

**La**

**t**

$

𝑢

\)

r

𝑧" 

$

**h **

$

%

**Cl**

𝑧

**La**

𝑧̂

**Co**

**GCN**

𝑢

$

**Text**

%

**ye**

%

𝑢& 

\! 𝑢

𝑢\#

\(

**a**

𝑓

**r**

**ye**

**s**

Au

\(O

\#

**s**

**r**

**n**

𝑢' 

𝑢" 

**s**

***Random Mask***

**i**

**Excited**

$

**c**

**f**

***Audio***

p

d

**i**

e

i

𝑧

**a**

**c**

o

n

\#

**a**



𝑥

**\(B1\) Global Contextual Fusion**

**t**

**Initialize**

-

E

" 

**e**

**t**

Sm

**i**

n

**n**

**o**

c

**n**

i

o

**a**

l

****

e

d

**Se**

**t**

**L**

**Happy**

\)

er

& 

**Visual **Project to Shared

**io**

𝑢$

𝑢

**a**

𝑓\! 

**l**

" 

**f**

%

**y**

W

Hidden Space

****

**n**

\! 

**At**

**GCN**

𝑢\! 

**e**

& 

𝑢

𝑢

\#

**r**

𝑧

& 

𝑢

\(S

**t**

\(

Te

\! 

**ent**

***Global Proximity***

𝑢' 

𝑢" 

**Neutral**

e

𝑧

n

x

**Audio**

***Text***

t

& 

**i**

e

t 

𝑓

**o**

n



" 

E

" 

**n **

c

W

Don’t you do that. 

e

n

𝑥\#

& 

\#

**L**

***\(C1\) Graph Construction***

***\(C3\) Contrastive Learning***

-

c

𝑧

Don’t you say goodbyes. 

BE

o

" 

**a**

d

**y**

𝑧̂&% 

**Sad**

R

e

& 

**Text**

𝑥

**e**

T

r

𝑓

**r**

𝑦

\)

\#

W"$

***\(C2\) Graph Augmentation***

𝑧&\# **\(B2\) Specific Modalities Fusion**

Figure 2: Overview of JOYFUL. We first extract uni-modal features, fuse them using a multimodal fusion module, and use them as input of the GCL-based framework to learn better representations for emotion recognition. 

2022\). By capturing the interactions between differ-able graph augmentation\) and supervised \(cross-

ent modalities more reasonably, deep models can

entropy\) manners to fully explore graph structural

acquire more comprehensive information. Current

information and downstream supervisory signals. 

fusion methods can be classified into aggregation-

based \(Wu et al., 2021; Guo et al., 2021\), alignment-3 Methodology

based \(Liu et al., 2020; Li et al., 2022e\), and their mixture \(Wei et al., 2019; Nagrani et al., 2021\). 

Figure 2 shows an overview of JOYFUL, which Aggregation-based fusion methods \(Zadeh et al., 

mainly consists of four components: \(A\) a uni-

2017; Chen et al., 2021\) adopt concatenation, ten-modal extractor, \(B\) a multimodal fusion \(MF\)

sor fusion and memory fusion to combine multi-

module, \(C\) a graph contrastive learning module, 

ple modalities. Alignment-based fusion centers

and \(D\) a classifier. Hereafter, we give formal

on latent cross-modal adaptation, which adapts

notations and the task definition of JOYFUL, and

streams from one modality to another \(Wang et al., 

introduce each component subsequently in detail. 

2022a\). Different from the above methods, we 3.1 Notations and Task Definition

learn global contextual information by concatena-

tion while fully exploring the specific patterns of

In dialogue emotion recognition, a training dataset

each modality in an alignment manner. 

D = \{\(Ci, Yi\)\}N is given, where

i=1

Ci represents

the i-th conversation, each conversation contains

2.3 Graph Contrastive Learning

several utterances Ci = \{u1, . . . , um\}, and Yi ∈

GCL aims to learn representations by maximizing

Ym, given label set Y = \{y1, . . . , yk\} of k emo-

feature consistency under differently augmented

tion classes. Let Xv, Xa, Xt be the visual, audio, 

views, that exploit data- or task-specific augmenta-

and text feature spaces, respectively. The goal of

tions, to inject the desired feature invariance \(You

MERC is to learn a function F : Xv × Xa × Xt →

et al., 2020\). GCL has been well used in the NLP

Y that can recognize the emotion label for each

community via self-supervised and supervised set-

utterance. We utilize three widely used multimodal

tings. Self-supervised GCL first creates augmented

conversational benchmark datasets, namely IEMO-

graphs by edge/node deletion and insertion \(Zeng

CAP, MOSEI, and MELD, to evaluate the perfor-

and Xie, 2021\), or attribute masking \(Zhang et al., 

mance of our model. Please see Section 4.1 for

2022\). It then captures the intrinsic patterns and their detailed statistical information. 

properties in the augmented graphs without using

human provided labels. Supervised GCL designs

3.2 Uni-modal Extractor

adversarial \(Sun et al., 2022\) or geometric \(Li et al., 

For IEMOCAP \(Busso et al., 2008\), video features

2022d\) contrastive loss to make full use of label in-xv ∈ R512, audio features xa ∈ R100, and text fea-

formation. For example, Li et al. \(2022c\) first used tures xt ∈ R768 are obtained from OpenFace \(Bal-

supervised CL for emotion recognition, greatly im-

trusaitis et al., 2018\), OpenSmile \(Eyben et al., 

proving the performance. Inspired by previous

2010\) and SBERT \(Reimers and Gurevych, 2019\), 

studies, we jointly consider self-supervised \(suit-

respectively. For MELD \(Poria et al., 2019a\), xv ∈

16053

R342, xa ∈ R300, and xt ∈ R768 are obtained contains essential modality cues for downstream

from DenseNet \(Huang et al., 2017\), OpenSmile, emotion recognition, we reconstruct zgm from ˆzgm

and TextCNN \(Kim, 2014\). For MOSEI \(Zadeh

by minimizing their Euclidean distance:

et al., 2018\), xv ∈ R35, xa ∈ R80, and xt ∈ R768

are obtained from TBJE \(Delbrouck et al., 2020\), 

J grec = ∥ˆ

zgm − zgm∥2. 

\(2\)

LibROSA \(Raguraman et al., 2019\), and SBERT. 

Textual features are sentence-level static features. 

3.3.2 Specific Representation Learning

Audio and visual modalities are utterance-level fea-

Specific representation learning aims to fully ex-

tures by averaging all the token features. 

plore specific information from each modality to

complement one another. Figure 2 \(B2\) shows that 3.3 Multimodal Fusion Module

we first use three fully connected deep neural net-

Though the uni-modal extractors can capture long-

works fℓ

\(·\) to project uni-modal embeddings

term temporal context, they are unable to handle

\{v,a,t\}

x\{v,a,t\} into a hidden space with representations as

feature redundancy and noise due to the modality

zℓ

. Considering that visual, audio, and text

gap. Thus, we design a new multimodal fusion

\{v,a,t\}

features are extracted with different encoding meth-

module \(Figure 2 \(B\)\) to inherently separate mul-ods, directly applying multiple specific features as

tiple modalities into two disjoint parts, contextual

an input for the downstream emotion recognition

representations and specific representations, to ex-

task will degrade the model’s accuracy. To solve it, 

tract the consistency and specificity of heteroge-

the multimodal features are projected into a shared

neous modalities collaboratively and individually. 

subspace, and a shared trainable basis matrix is

3.3.1 Contextual Representation Learning

designed to learn aligned representations for them. 

Therefore, the multimodal features can be fully

Contextual representation learning aims to explore

integrated and interacted to mitigate feature discon-

and learn hidden contextual intent/topic knowledge

tinuity and remove noise across modalities. We

of the dialogue, which can greatly improve the per-

define a shared trainable basis matrix B with q ba-

formance of JOYFUL. In Figure 2 \(B1\), we first sis vectors as B = \(b

project all uni-modal inputs

1, . . . , bq\)T ∈ Rq×db with db

x\{v,a,t\} into a latent

representing the dimensionality of each basis vec-

space by using three separate connected deep neu-

tor. Here, T indicates transposition. Then, zℓ

ral networks fg

\(

\{v,a,t\}

\{

·\) to obtain hidden represen-

v,a,t\}

and B are projected into the shared subspace:

tations zg

. Then, we concatenate them as zg

\{v,a,t\}

m

and apply it to a multi-layer transformer to maxi-

˜

zℓ

mize the correlation between multimodal features, 

\{v,a,t\} = W\{v,a,t\}zℓ\{v,a,t\}, 

e

B = BWb, \(3\)

where we learn a global contextual multimodal rep-

where W\{v,a,t,b\} are trainable parameters. To learn

resentation ˆ

zgm. Considering that the contextual

new representations for each modality, we calculate

information will change over time, we design a

the cosine similarity between them and B as

temporal smoothing strategy for ˆ

zgm as

S\{v,a,t\} = \( ˜

zℓ

ij

J

\{v,a,t\}\)i · e

bj, 

\(4\)

smooth = ∥ ˆ

zgm − zcon∥2, 

\(1\)

where Sv denotes the similarity between the i-th

where

ij

zcon is the topic-related vector describing the

visual feature \(˜

zℓ

high-level global contextual information without

v\)i and the j-th basis vector repre-

sentation eb

requiring topic-related inputs, following the defini-

j . To prevent inaccurate representation

learning caused by an excessive weight of a certain

tion in Joshi et al. \(2022\). We update the \(i\+1\)-th item, the similarities are further normalized by

utterance as zcon ← zcon \+eη∗i ˆzgm, and η is the ex-

ponential smoothing parameter \(Shazeer and Stern, 

2018\), indicating that more recent information will exp \(S\{v,a,t\}\)

S\{v,a,t\} =

ij

. 

\(5\)

ij

P

be more important. 

q

exp \(S\{v,a,t\}\)

k=1

ik

To ensure fused contextual representations cap-

ture enough details from hidden layers, Hazarika

Then, the new representations are obtained as

et al. \(2020\) minimized the reconstruction error be-q

tween fused representations with hidden represen-

X

\( ˆ

zℓ

S\{v,a,t\}

tations. Inspired by their work, to ensure that ˆ

zg

\{v,a,t\}\)i =

ik

· ebk, 

\(6\)

m

k=1

16054





where ˆ

zℓ

are new representations, and we also

provide highly heterogeneous contexts for nodes. 

\{v,a,t\}

use reconstruction loss for their combinations

By maximizing the mutual information between

two augmented views, we can improve the robust-

J ℓrec = ∥ˆ

zℓm − zℓm∥2, 

\(7\)

ness of the model and obtain distinguishable node

representations \(You et al., 2020\). However, there where Concat\( , \) indicating the concatenation, i.e., 

are no universally appropriate GA methods for var-

ˆ

zℓ =Concat

= Concat

m

\( ˆ

zℓv, ˆ

zℓa, ˆ

zℓt\), zℓm

\(zℓv, zℓa, zℓt\). 

ious downstream tasks \(Xu et al., 2021\), which Finally, we define the multimodal fusion loss by

motivates us to design specific GA strategies for

combining Eqs.\(1\), \(2\), and \(7\) as: MERC. Considering that MERC is sensitive to ini-L

tialized representations of utterances, intra-speaker

mf = Jsmooth \+ J g

rec \+ J ℓ

rec. 

\(8\)

and inter-speaker dependencies, we design three

3.4 Graph Contrastive Learning Module

corresponding GA methods:

- Feature Masking \(FM\): given the initialized

Intra dependency

First-order Inter dependency

Second-order Inter dependency

Speaker-A

Speaker-A

representations of utterances, we randomly

Speaker-B

𝑢' 

Speaker-B

' 

\(

𝑢\(

select p dimensions of the initialized repre-

𝑢" 

" 

\! 

𝑢\! 

sentations and mask their elements with zero, 

𝑢' 

𝑢' 

𝑢" 

\)

" 

\)

which is expected to enhance the robustness

$

𝑢$

of JOYFUL to multimodal feature variations; 

𝑢' 

𝑢' 

𝑢" 

& 

" 

& 

%

𝑢%

- Edge Perturbation \(EP\): given the graph G, 

𝑢" 

" 

\#

**Window size=1**

𝑢\#

**Window size=2**

we randomly drop and add p% of intra- and

inter-speaker edges, which is expected to en-

Figure 3: An example of graph construction. 

hance the robustness of JOYFUL to local struc-

tural variations; 

3.4.1 Graph Construction

-

Graph construction aims to establish relations be-

Global Proximity \(GP\): given the graph G, 

we first use the Katz index \(Katz, 1953\) to cal-tween past and future utterances that preserve both

culate high-order similarity between intra- and

intra- and inter-speaker dependencies in a dialogue. 

inter-speakers, and randomly add

We define the

p% high-

i-th dialogue with P speakers as Ci =

order edges between speakers, which is ex-

\{US1, . . . , USP \}, where USi = \{uSi

1 , . . . , uSi

m \} rep-

pected to enhance the robustness of J

resents the set of utterances spoken by speaker

OYFUL

Si. 

to global structural variations \(Examples in

Following Ghosal et al. \(2019\), we define a graph Appendix A\). 

with nodes representing utterances and directed

edges representing their relations: Rij = ui → uj, 

We propose a hybrid scheme for generating

where the arrow represents the speaking order. 

graph views on both structure and attribute levels

Intra-Dependency \(Rintra ∈ \{USi → USi\}\) rep-

to provide diverse node contexts for the contrastive

resents intra-relations between the utterances \(red

objective. Figure 2 \(C\) shows that the combina-lines\), and Inter-Dependency \(Rinter ∈ \{USi →

tion of \(FM & EP\) and \(FM & GP\) are adopted to

USj \}, i ̸= j\) represents the inter-relations between

obtain two correlated views. 

the utterances \(purple lines\), as shown in Figure 3. 

All nodes are initialized by concatenating contex-

3.4.3 Graph Contrastive Learning

tual and specific representations as hm = Con-

Graph contrastive learning adopts an L-th layer

cat\(ˆ

zgm, ˆ

zℓ \). And we show that window size is a

m

GCNs as a graph encoder to extract node hidden

hyper-parameter that controls the context informa-

tion for each utterance and provide accurate intra-

representations H\(1\) = \{h\(1\), 

1

. . . , h\(1\)

m \} and H\(2\)

and inter-speaker dependencies. 

= \{h\(2\), 

1

. . . , h\(2\)

m \} for two augmented graphs, 

where hi is the hidden representation for the i-th

3.4.2 Graph Augmentation

node. We follow an iterative neighborhood aggre-

Graph Augmentation \(GA\): Inspired by Zhu et al. 

gation \(or message passing\) scheme to capture the

\(2020\), creating two augmented views by using structural information within the nodes’ neighbor-different ways to corrupt the original graph can

hood. Formally, the propagation and aggregation

16055

of the ℓ-th GCN layer is:

Dataset

Train

Valid

Test

IEMOCAP\(4-way\)

3,200/108

400/12

943/31

a\(i, ℓ\) = AGG\(ℓ\) \(\{h\(j, ℓ−1\)|j ∈ Ni\}\)

\(9\)

IEMOCAP\(6-way\)

5,146/108

664/12

1,623/31

h

MELD

9,989/1,039 1,109/114 2,80/2,610

\(i, ℓ\) = COM\(ℓ\) \(h\(i, ℓ−1\) ⊕ a\(i, ℓ\)\), 

\(10\)

MOSEI

16,327/2,249 1,871/300 4,662/679

where h\(i,ℓ\) is the embedding of the i-th node at

Table 1: Utterances/Conversations of four datasets. 

the ℓ-th layer, h\(i,0\) is the initialization of the i-

th utterance, Ni represents all neighbour nodes

of the i-th node, and AGG

where

\(ℓ\)\(·\) and COM\(ℓ\)\(·\)

k is the number of emotion classes, m is the

are aggregation and combination of the ℓ-th GCN

number of utterances, ˆyj is the i-th predicted label, 

i

layer \(Hamilton et al., 2017\). For convenience, we and yj is the i-th ground truth of j-th class. 

i

define hi = h\(i,L\). After the L-th GCN layer, final

Above all, combining the MF loss of Eq.\(8\), 

node representations of two views are H\(1\) / H\(2\). 

contrastive loss of Eq.\(13\), and classification loss In Figure 2 \(C3\), we design the intra- and inter-of Eq.\(14\) together, the final objective function is view graph contrastive losses to learn distinctive

node representations. We start with the inter-view

Lall = αLmf \+ βLct \+ Lce, 

\(15\)

contrastiveness, which pulls closer the representa-

where α and β are the trade-off hyper-parameters. 

tions of the same nodes in two augmented views

We give our pseudo-code in Appendix F. 

while pushing other nodes away, as depicted by the

red and blue dash lines in Figure 2 \(C3\). Given 4 Experiments and Result Analysis

the definition of our positive and negative pairs as

4.1 Experimental Settings

\(h\(1\), h\(2\)\)\+ and \(h\(1\), h\(2\)\)−, where i

i

i

i

j

̸= j, the

inter-view loss for the i-th node is formulated as:

Datasets and Metrics. In Table 1, IEMOCAP

is a conversational dataset where each utterance

exp\(sim\(h\(1\), h\(2\)\)\)

was labeled with one of the six emotion categories

Li

i

i

inter = − log

, 

m

P

\(11\)

\(Anger, Excited, Sadness, Happiness, Frustrated

exp\(sim\(h\(1\), h\(2\)\)\)

i

j

and Neutral\). Following COGMEN, two IEMO-

j=1

CAP settings were used for testing, one with four

where sim\(·, ·\) denotes the similarity between two

emotions \(Anger, Sadness, Happiness and Neu-

vectors, i.e., the cosine similarity in this paper. 

tral\) and one with all six emotions, where 4-way

Intra-view contrastiveness regards all nodes ex-

directly removes the additional two emotion la-

cept the anchor node as negatives within a partic-

bels \(Excited and Frustrated\). MOSEI was labeled

ular view \(green dash lines in Figure 2 \(C3\)\), as with six emotion labels \(Anger, Disgust, Fear, Hap-defined \(h\(1\), h\(1\)\)− where i

piness, Sadness, and Surprise\). For six emotion

i

j

̸= j. The intra-view

contrastive loss for the i-th node is defined as:

labels, we conducted two settings: binary classi-

fication considers the target emotion as one class

exp\(sim\(h\(1\), h\(2\)\)\)

Li

i

i

and all other emotions as another class, and multi-

intra = − log

. 

m

P

\(12\)

exp\(sim\(h\(1\), h\(1\)\)\)

label classification tags multiple labels for each

i

j

j=1

utterance. MELD was labeled with six universal

emotions \(Joy, Sadness, Fear, Anger, Surprise, and

By combining the inter- and intra-view con-

Disgust\). We split the datasets into 70%/10%/20%

trastive losses of Eqs.\(11\) and \(12\), the contrastive as training/validation/test data, respectively. Fol-objective function Lct is formulated as:

lowing Joshi et al. \(2022\), we used Accuracy and m

Weighted F1-score \(WF1\) as evaluation metrics. 

1 X

Lct =

\(Li

Please note that the detailed label distribution of

2m

inter \+ Liintra\). 

\(13\)

i=1

the datasets is given in Appendix I. 

3.5 Emotion Recognition Classifier

Implementation Details. We selected the aug-

mentation pairs \(FM & EP\) and \(FM & GP\) for

We use cross-entropy loss for classification as:

two views. We set the augmentation ratio p=20%

and smoothing parameter

m

k

η=0.2, and applied the

1 X X

L

Adam \(Kingma and Ba, 2015\) optimizer with an ce = −

yj log \(ˆ

yj\), 

\(14\)

m

i

i

initial learning rate of 3

i=1 j=1

e-5. For a fair comparison, 

16056

we followed the default parameter settings of the Method

IEMOCAP 6-way \(F1\) ↑

Average ↑

baselines and repeated all experiments ten times to

Hap. 

Sad. Neu. Ang. Exc. Fru. 

Acc. 

WF1

report the average accuracy. We conducted the sig-

Mult

48.23

76.54 52.38 60.04 54.71 57.51

58.04

58.10

nificance by t-test with Benjamini-Hochberg \(Ben-

FE2E

44.82

64.98 56.09 62.12 61.02 57.14

58.30

57.69

DiaRNN

32.88

78.08 59.11 63.38 73.66 59.41

63.34

62.85

jamini and Hochberg, 1995\) correction \(Please see COSMIC

53.23

78.43 62.08 65.87 69.60 61.39

64.88

65.38

Af-CAN

37.01

72.13 60.72 67.34 66.51 66.13

64.62

63.74

details in Appendix G\). 

AGHMN

52.10

73.30 58.40 61.91 69.72 62.31

63.58

63.54

Baselines. Different MERC datasets have dif-

RGAT

51.62

77.32 65.42 63.01 67.95 61.23

65.55

65.22

COGMEN 51.91

81.72 68.61 66.02 75.31 58.23

68.26

67.63

ferent best system results, following COGMEN, 

JOYFUL

60.94† 84.42† 68.24 69.95† 73.54 67.55† 70.55† 71.03†

we selected SOTA baselines for each dataset. 

For IEMOCAP-4, we selected Mult \(Tsai et al., 

Table 2: Overall performance comparison on IEMO-

2019a\), RAVEN \(Wang et al., 2019\), MTAG \(Yang

CAP \(6-way\) in the multimodal \(A\+T\+V\) setting. Sym-

et al., 2021\), PMR \(Lv et al., 2021\), COG-bol † indicates that JOYFUL significantly surpassed all

baselines using t-test with

MEN and MICA \(Liang et al., 2021\) as our p < 0.005. 

baselines. For IEMOCAP-6, we selected Mult, 

FE2E \(Dai et al., 2021\), DiaRNN \(Majumder

Method

Happy

Sadness

Neutral

Anger

WF1

et al., 2019\), COSMIC \(Ghosal et al., 2020\), Af-Mult

88.4

86.3

70.5

87.3

80.4

RAVEN

86.2

83.2

69.4

86.5

78.6

CAN \(Wang et al., 2021\), AGHMN \(Jiao et al., 

MTAG

85.9

80.1

64.2

76.8

73.9

2020\), COGMEN and RGAT \(Ishiwatari et al., 

PMR

89.2

87.1

71.3

87.3

81.0

MICA

83.7

75.5

61.8

72.6

70.7

2020\) as our baselines. For MELD, we selected COGMEN

78.8

86.8

84.6

88.0

84.9

DiaGCN \(Ghosal et al., 2019\), DiaCRN \(Hu et al., 

JOYFUL

80.1

88.1†

85.1†

88.1†

85.7†

2021\), MMGCN \(Wei et al., 2019\), UniMSE \(Hu

Table 3: Overall performance comparison on IEMO-

et al., 2022b\), COGMEN and MM-DFN \(Hu et al., 

CAP \(4-way\) in the multimodal \(A\+T\+V\) setting. 

2022a\) as baselines. For MOSEI, we selected Mul-Net \(Shenoy et al., 2020\), TBJE \(Delbrouck et al., 

2020\), COGMEN and MR \(Tsai et al., 2020\). 

4.3 Performance of JOYFUL

4.2 Parameter Sensitive Study

Tables 2 & 3 show that JOYFUL outperformed all baselines in terms of accuracy and WF1, improving

We first examined whether applying different data

5.0% and 1.3% in WF1 for 6-way and 4-way, re-

augmentation methods improves JOYFUL. We ob-

spectively. Graph-based methods, COGMEN and

served in Figure 4 \(A\) that 1\) all data augmentation JOYFUL, outperform Transformers-based methods, 

strategies are effective 2\) applying augmentation

Mult and FE2E. Transformers-based methods can-

pairs of the same type cannot result in the best per-

not distinguish intra- and inter-speaker dependen-

formance; and 3\) applying augmentation pairs of

cies, distracting their attention to important utter-

different types improves performance. Thus, we

ances. Furthermore, they use the cross-modal at-

selected \(FM & EP\) and \(FM & GP\) as the default

tention layer, which can enhance common features

augmentation strategy since they achieved the best

among modalities while losing uni-modal specific

performance \(More details please see Appendix C\). 

features \(Rajan et al., 2022\). JOYFUL outperforms JOYFUL has three hyperparameters. α and β de-other GNN-based methods since it explored fea-

termine the importance of MF and GCL in Eq.\(15\), 

tures from both the contextual and specific levels, 

and window size controls the contextual length of

and used GCL to obtain more distinguishable fea-

conversations. In Figure 4 \(B\), we observed how α

tures. However, JOYFUL cannot improve in Happy

and β affect the performance of JOYFUL by varying

for 4-way and in Excited for 6-way since samples

α from 0.02 to 0.10 in 0.02 intervals and β from

in IEMOCAP were insufficient for distinguishing

0.1 to 0.5 in 0.1 intervals. The results indicated

these similar emotions \(Happy is 1/3 of Neutral in

that JOYFUL achieved the best performance when

Fig. 4 \(D\)\). Without labels’ guidance to re-sample α ∈ \[0.06, 0.08\] and β = 0.3. Figure 4 \(C\) shows or re-weight the underrepresented samples, self-that when window\_size = 8, JOYFUL achieved the

supervised GCL, utilized in JOYFUL, cannot en-

best performance. A small window size will miss

sure distinguishable representations for samples of

much contextual information, and a longer one con-

minor classes by only exploring graph topological

tains too much noise, we set it as 8 in experiments

information and vertex attributes. 

\(Details in Appendix D\). 

Tables 4 & 5 show that JOYFUL outperformed 16057





**0.86**

**e**

**e**

**e**

**86**

**0.85**

**84**

**82**

**80**

**0.84**

**78**

**76**

**0.83**

**of samples**

**eighted F1 Scor**

**eighted F1 Scor 0.1**

**eighted F1 Scor**

**edicted Label**

**0.2**

**W**

**W**

**W 0.82**

**Pr**

**0.3**

**0.080.10**

**ȕ**

**0.4**

**0.06**

**Number**

**0.5**

**0.020.04**

**ܤ**

**0.81**

**1 2 3 4 5 6 7 8 9 10 11**

**True Label**

**\(A\) IEMOCAP \(4-way\) Classification **

**\(B\) IEMOCAP \(4-way\) Classification **

**\(C\) IEMOCAP \(4-way\) Window Size**

**\(D\) IEMOCAP \(4-way\) Error Visualization**

Figure 4: \(A\) WF1 gain with different augmentation pairs; \(B∼C\) Parameter tuning; \(D\) Imbalanced dataset. 

Methods

Emotion Categories of MELD \(F1\) ↑

Average ↑

Modality

IEMOCAP-4 IEMOCAP-6

MOSEI \(WF1\)

Neu. 

Sur. 

Sad. 

Joy

Anger

Acc. 

WF1

Acc. 

WF1

Acc. 

WF1

Binary Multi-label

DiaGCN

75.97 46.05

19.60

51.20

40.83

58.62

56.36

Audio

64.8

63.3

49.2

48.0

51.2

53.3

DiaCRN

77.01 50.10

26.63

52.77

45.15

61.11

58.67

Text

83.0

83.0

67.4

67.5

73.6

73.9

MMGCN 76.33 48.15

26.74

53.02

46.09

60.42

58.31

Video

44.6

43.4

28.2

28.6

23.6

24.4

UniMSE

74.61 48.21

31.15

54.04

45.26

59.39

58.19

A\+T

82.6

82.5

67.5

67.8

74.7

74.9

COGMEN 75.31 46.75

33.52

54.98

45.81

58.35

58.66

A\+V

68.0

67.5

52.7

52.5

61.7

62.4

MM-DFN 77.76 50.69

22.93

54.78

47.82

62.49

59.46

T\+V

80.0

80.0

65.2

65.5

73.1

73.4

JOYFUL

76.80 51.91† 41.78† 56.89† 50.71† 62.53† 61.77†

w/o MF\(B1\)

85.3

85.4

70.0

70.3

76.2

76.5

w/o MF\(B2\)

85.2

85.1

69.2

69.5

75.8

76.2

w/o MF

85.2

84.9

69.0

69.2

75.4

75.8

Table 4: Results on MELD with the multimodal setting. 

COGMEN w/o GNN 80.1

80.2

62.7

62.9

72.3

72.9

Underline indicates our reproduced results. 

w/o GCL

84.7

84.7

66.1

66.5

73.8

73.4

JOYFUL

85.6† 85.7† 70.5† 71.0†

76.9†

77.2†

Method

Happy

Sadness Anger

Fear

Disgust Surprise

Table 6: Ablation study with different modalities. 

Binary Classification \(F1\) ↑

Mul-Net

67.9

65.5

67.2

87.6

74.7

86.0

TBJE

63.8

68.0

74.9

84.1

83.8

86.1

separately improve the performance of JOYFUL, 

MR

65.9

66.7

71.0

85.9

80.4

85.9

COGMEN

70.4

72.3

76.2

88.1

83.7

85.3

showing their effectiveness \(Visualization in Ap-

JOYFUL

71.7†

73.4†

78.9†

88.2

85.1†

86.1

pendix H\). JOYFUL w/o GCL and COGMEN w/o Multi-label Classification \(F1\) ↑

GNN utilize only a multimodal fusion mechanism

Mul-Net

70.8

70.9

74.5

86.2

83.6

87.7

TBJE

68.4

73.9

74.4

86.3

83.1

86.6

for classification without additional modules for

MR

69.6

72.2

72.8

86.5

82.5

87.9

optimizing node representations. The comparison

COGMEN

72.7

73.9

78.0

86.7

85.5

88.3

JOYFUL

70.9

74.6†

78.1†

89.4†

86.8†

90.5†

between them demonstrates the effectiveness of the

multimodal fusion mechanism in JOYFUL. 

Table 5: Results on MOSEI with the multimodal setting. 

Method

One-Layer \(WF1\) Two-Layer \(WF1\) Four-Layer \(WF1\)

COGMEN JOYFUL COGMEN JOYFUL COGMEN JOYFUL

the baselines in more complex scenes with multiple

speakers or various emotional labels. Compared

Unattack

67.63

71.03

63.21

71.05

58.39

70.96

with COGMEN and MM-DFN, which directly ag-

5% Noisy

65.26

70.82

61.35

70.55

56.28

70.10

10% Noisy

62.26

70.33

59.24

70.45

53.21

69.23

gregate multimodal features, JOYFUL can fully ex-

15% Noisy

57.28

69.98

55.18

69.21

52.32

67.96

20% Noisy

54.22

68.52

51.79

68.82

50.72

67.23

plore features from each uni-modality by specific

representation learning to improve the performance. 

Table 7: Adversarial attacks for GNN with different

The GCL module can better aggregate similar emo-

depth on 6-way IEMOCAP. 

tional features for utterances to obtain better per-

formance for multi-label classification. We cannot

We deepened the GNN layers to verify JOYFUL’s

improve in Happy on MOSEI since the samples are

ability to alleviate the over-smoothing. In Table 7, 

imbalanced and Happy has only 1/6 of Surprise, 

COGMEN with four-layer GNN was 9.24% lower

making JOYFUL hard to identify it. 

than that with one-layer, demonstrating that the

To verify the performance gain from each com-

over-smoothing decreases performance, while JOY-

ponent, we conducted additional ablation studies. 

FUL relieved this issue by using the GCL frame-

Table 6 shows multi-modalities can greatly improve work. To verify the robustness, following Tan et al. 

JOYFUL’s performance compared with each single

\(2022\), we randomly added 5%∼20% noisy edges modality. GCL and each component of MF can

to the training data. In Table 7, COGMEN was 16058





Figure 5: t-SNE visualization of IEMOCAP \(6-way\). 

***I just miss him. \(Sad\)***

For future work, we plan to investigate the perfor-

mance of using supervised GCL for JOYFUL on

***It does look really beautiful over the water. \(Happy\) ***

unbalanced and small-scale emotional datasets. 

***Oh, thanks. move here before you get married. \(Excited\)***

## Acknowledgements

***0aybe we can find you something with juggling. \(Neutral\)***

***You above all have got to believe. \(Angry\)***

The authors would like to thank Ying Zhang 1 for her advice and assistance. We gratefully acknowl-

***So what, I'm not fast with women. \(Frustrated\)***

edge anonymous reviewers for their helpful com-

ments and feedback. We also acknowledge the

Figure 6: Visualization of emotion probability, each first authors of COGMEN \(Joshi et al., 2022\): Abhinav row is JOYFUL and each second row is COGMEN. 

Joshi and Ashutosh Modi for sharing codes and

datasets. Finally, Dongyuan Li acknowledges the

easily affected by the noise, decreasing 10.8% per-

support of the China Scholarship Council \(CSC\). 

formance in average with 20% noisy edges, while

J

Limitations

OYFUL had strong robustness with only an average

2.8% performance reduction for 20% noisy edges. 

JOYFUL has a limited ability to classify minority

To show the distinguishability of the node repre-

classes with fewer samples in unbalanced datasets. 

sentations, we visualize the node representations of

Although we utilized self-supervised graph con-

FE2E, COGMEN, and JOYFUL on 6-way IEMO-

trastive learning to learn a distinguishable repre-

CAP. In Figure 5, COGMEN and JOYFUL obtained sentation for each utterance by exploring vertex

more distinguishable node representations than

attributes, graph structure, and contextual infor-

FE2E, demonstrating that graph structure is more

mation, GCL failed to separate classes with fewer

\(A\) Initialized Features

\(B\) Output Features

suitable for MERC than Transformers. JOYFUL

samples from the ones with more samples because

performed better than COGMEN, illustrating the ef-

the utilized self-supervised learning lacks the label

fectiveness of GCL. In Figure 6, we randomly sam-information and does not balance the label distribu-

pled one example from each emotion of IEMOCAP

tion. Another limitation of JOYFUL is that its frame-

\(6-way\) and chose best-performing COGMEN for

work was designed specifically for multimodal

comparison. JOYFUL obtained more discriminate

emotion recognition tasks, which is not straight-

prediction scores among emotion classes, show-

forward and general as language models \(Devlin

ing GCL can push samples from different emotion

et al., 2019; Liu et al., 2019\) or image processing class farther apart. 

techniques \(LeCun et al., 1995\). This setting may limit the applications of JOYFUL for other mul-5 Conclusion

timodal tasks, such as the multimodal sentiment

We proposed a joint learning model \(JOYFUL\) for

analysis task \(Detailed experiments in Appendix J\)

MERC, that involves a new multimodal fusion

and the multimodal retrieval task. Finally, although

mechanism and GCL module to effectively im-

JOYFUL achieved SOTA performances on three

prove the performance of MERC. The MR mecha-

widely-used MERC benchmark datasets, its per-

nism can extract and fuse contextual and uni-modal

formance on larger-scale and more heterogeneous

specific emotion features, and the GCL module

data in real-world scenarios is still unclear. 

can help learn more distinguishable representations. 

1scholar.google.com/citations?user=tbDNsHs

16059

References

Deepanway Ghosal, Navonil Majumder, Soujanya Poria, 

Niyati Chhaya, and Alexander Gelbukh. 2019. Di-

Tadas Baltrusaitis, Amir Zadeh, Yao Chong Lim, and

alogueGCN: A graph convolutional neural network

Louis-Philippe Morency. 2018. Openface 2.0: Facial

for emotion recognition in conversation. In Proc. of

behavior analysis toolkit. In Proc. of FG, pages 59–

EMNLP-IJCNLP, pages 154–164. 

66. 

Yoav Benjamini and Yosef Hochberg. 1995. Controlling

Xiaobao Guo, Adams Kong, Huan Zhou, Xianfeng

the false discovery rate: a practical and powerful

Wang, and Min Wang. 2021. Unimodal and cross-

approach to multiple testing. Journal of the Royal

modal refinement network for multimodal sequence

statistical society, 57\(1\):289–300. 

fusion. In Proc. of EMNLP, pages 9143–9153. 

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe

Vikram Gupta, Trisha Mittal, Puneet Mathur, Vaibhav

Kazemzadeh, Emily Mower, Samuel Kim, Jean-

Mishra, Mayank Maheshwari, Aniket Bera, Debdoot

nette N. Chang, Sungbok Lee, and Shrikanth S. 

Mukherjee, and Dinesh Manocha. 2022. 3massiv:

Narayanan. 2008. IEMOCAP: interactive emotional

Multilingual, multimodal and multi-aspect dataset of

dyadic motion capture database. Lang. Resour. Eval-

social media short videos. In Proc. of CVPR, pages uation, 42\(4\):335–359. 

21032–21043. 

Zhihong Chen, Yaling Shen, Yan Song, and Xiang Wan. 

William L. Hamilton, Zhitao Ying, and Jure Leskovec. 

2021. Cross-modal memory networks for radiology

2017. Inductive representation learning on large

report generation. In Proc. of ACL/IJCNLP, pages

graphs. In Proc. of NeurIPS, pages 1024–1034. 

5904–5914. 

Wei Han, Hui Chen, Alexander F. Gelbukh, Amir

Wenliang Dai, Samuel Cahyawijaya, Zihan Liu, and

Zadeh, Louis-Philippe Morency, and Soujanya Poria. 

Pascale Fung. 2021. Multimodal end-to-end sparse

2021a. Bi-bimodal modality fusion for correlation-

model for emotion recognition. In Proc. of NAACL-

controlled multimodal sentiment analysis. In Proc. 

HLT, pages 5305–5316. 

of ICMI, pages 6–15. 

Jean-Benoit Delbrouck, Noé Tits, Mathilde Brousmiche, 

Wei Han, Hui Chen, and Soujanya Poria. 2021b. Im-

and Stéphane Dupont. 2020. A transformer-based

proving multimodal fusion with hierarchical mutual

joint-encoding for emotion recognition and sentiment

information maximization for multimodal sentiment

analysis. In Workshop on Multimodal Language

analysis. In Proc. of EMNLP, pages 9180–9192. 

\(Challenge-HML\), pages 1–7. 

Devamanyu Hazarika, Roger Zimmermann, and Sou-

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and

janya Poria. 2020. Misa: Modality-invariant and

Kristina Toutanova. 2019. BERT: pre-training of

-specific representations for multimodal sentiment

deep bidirectional transformers for language under-

analysis. In Proc. of MM, page 1122–1131. 

standing. In Proc. of NAACL-HLT, pages 4171–4186. 

Dou Hu, Xiaolong Hou, Lingwei Wei, Lian-Xin Jiang, 

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi

and Yang Mo. 2022a. MM-DFN: multimodal dy-

Reichart. 2018. The hitchhiker’s guide to testing

namic fusion network for emotion recognition in con-

statistical significance in natural language processing. 

versations. In Proc. of ICASSP, pages 7037–7041. 

In Proc. of ACL, pages 1383–1392. 

Florian Eyben, Martin Wöllmer, and Björn Schuller. 

Dou Hu, Lingwei Wei, and Xiaoyong Huai. 2021. Dia-

2010. Opensmile: The munich versatile and fast

logueCRN: Contextual reasoning networks for emo-

open-source audio feature extractor. In Proc. of MM, 

tion recognition in conversations. In Proc. of ACL, page 1459–1462. 

pages 7042–7052. 

Yahui Fu, Shogo Okada, Longbiao Wang, Lili Guo, 

Guimin Hu, Ting-En Lin, Yi Zhao, Guangming Lu, 

Yaodong Song, Jiaxing Liu, and Jianwu Dang. 

Yuchuan Wu, and Yongbin Li. 2022b. UniMSE:

2021. CONSK-GCN: conversational semantic- and

Towards unified multimodal sentiment analysis and

knowledge-oriented graph convolutional network for

emotion recognition. In Proc. of EMNLP, pages

multimodal emotion recognition. In Proc. of ICME, 7837–7851. 

pages 1–6. 

Gao Huang, Zhuang Liu, Laurens van der Maaten, and

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. 

Kilian Q. Weinberger. 2017. Densely connected con-

Simcse: Simple contrastive learning of sentence em-

volutional networks. In Proc. of CVPR, pages 2261–

beddings. In Proc. of EMNLP, pages 6894–6910. 

2269. 

Deepanway Ghosal, Navonil Majumder, Alexander Gel-

Taichi Ishiwatari, Yuki Yasuda, Taro Miyazaki, and Jun

bukh, Rada Mihalcea, and Soujanya Poria. 2020. 

Goto. 2020. Relation-aware graph attention networks

COSMIC: COmmonSense knowledge for eMotion

with relational position encodings for emotion recog-

identification in conversations. 

In Findings of

nition in conversations. In Proc. of EMNLP, pages EMNLP, pages 2470–2481. 

7360–7370. 

16060

Wenxiang Jiao, Michael R. Lyu, and Irwin King. 2020. 

Tao Liang, Guosheng Lin, Lei Feng, Yan Zhang, and

Real-time emotion recognition via attention gated

Fengmao Lv. 2021. Attention is not enough: Miti-

hierarchical memory network. In Proc. of AAAI, 

gating the distribution discrepancy in asynchronous

pages 8002–8009. 

multimodal sequence fusion. In Proc. of ICCV, pages 8128–8136. 

Abhinav Joshi, Ashwani Bhat, Ayush Jain, Atin Singh, 

and Ashutosh Modi. 2022. COGMEN: COntextual-

Yunlong Liang, Fandong Meng, Jinan Xu, Jiaan Wang, 

ized GNN based multimodal emotion recognitioN. 

Yufeng Chen, and Jie Zhou. 2023. Summary-oriented

In Proc. of NAACL, pages 4148–4164. 

vision modeling for multimodal abstractive summa-

rization. In Proc. of ACL, pages 2934–2951. 

Leo Katz. 1953. A new status index derived from socio-

metric analysis. Psychometrika, 18:39–43. 

Yan Ling, Jianfei Yu, and Rui Xia. 2022. Vision-

language pre-training for multimodal aspect-based

Yoon Kim. 2014. Convolutional neural networks for

sentiment analysis. In Proc. of ACL, pages 2149–

sentence classification. In Proc. of EMNLP, pages 2159. 

1746–1751. 

Nayu Liu, Xian Sun, Hongfeng Yu, Wenkai Zhang, and

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A

Guangluan Xu. 2020. Multistage fusion with forget

method for stochastic optimization. In Proc. of ICLR, 

gate for multimodal summarization in open-domain

pages 1–15. 

videos. In Proc. of EMNLP, pages 1834–1845. 

Thomas N. Kipf and Max Welling. 2017. 

Semi-

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-

supervised classification with graph convolutional

dar Joshi, Danqi Chen, Omer Levy, Mike Lewis, 

networks. In Proc. of ICLR, pages 1–14. 

Luke Zettlemoyer, and Veselin Stoyanov. 2019. 

Roberta: A robustly optimized BERT pretraining

Hung Le, Nancy Chen, and Steven Hoi. 2022. Multi-

approach. CoRR, abs/1907.11692. 

modal dialogue state tracking. In Proc. of NAACL, pages 3394–3415. 

Fengmao Lv, Xiang Chen, Yanyong Huang, Lixin Duan, 

and Guosheng Lin. 2021. Progressive modality rein-

Yann LeCun, Yoshua Bengio, et al. 1995. Convolu-

forcement for human multimodal emotion recogni-

tional networks for images, speech, and time series. 

tion from unaligned multimodal sequences. In Proc. 

The handbook of brain theory and neural networks, 

of CVPR, pages 2554–2562. 

3361\(10\):1995. 

Navonil Majumder, Soujanya Poria, Devamanyu Haz-

Howard Levene et al. 1960. Contributions to probability

arika, Rada Mihalcea, Alexander F. Gelbukh, and

and statistics. Essays in honor of Harold Hotelling, Erik Cambria. 2019. Dialoguernn: An attentive RNN

278:292. 

for emotion detection in conversations. In Proc. of AAAI, pages 6818–6825. 

Dongyuan Li, Jingyi You, Kotaro Funakoshi, and Man-

abu Okumura. 2022a. A-TIP: attribute-aware text

Huisheng Mao, Ziqi Yuan, Hua Xu, Wenmeng Yu, Yihe

infilling via pre-trained language model. In Proc. of Liu, and Kai Gao. 2022. M-SENA: An integrated

COLING, pages 5857–5869. 

platform for multimodal sentiment analysis. In Proc. 

of ACL, pages 204–213. 

Sha Li, Madhi Namazifar, Di Jin, MOHIT BANSAL, 

Heng Ji, Yang Liu, and Dilek Hakkani-Tur. 2022b. 

Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, 

Enhanced knowledge selection for grounded dia-

Cordelia Schmid, and Chen Sun. 2021. Attention bot-

logues via document semantic graphs. In NAACL

tlenecks for multimodal fusion. In Proc. of NeurIPS, 2022. 

pages 14200–14213. 

Shimin Li, Hang Yan, and Xipeng Qiu. 2022c. Contrast

Aaron Nicolson, Jason Dowling, and Bevan Koopman. 

and generation make BART a good dialogue emotion

2023. e-health CSIRO at radsum23: Adapting a

recognizer. In Proc. of AAAI, pages 11002–11010. 

chest x-ray report generator to multimodal radiology

report summarisation. In The 22nd Workshop on Shuangli Li, Jingbo Zhou, Tong Xu, Dejing Dou, and

BioNLP@ACL, pages 545–549. 

Hui Xiong. 2022d. Geomgcl: Geometric graph con-

trastive learning for molecular property prediction. 

Timothy Ossowski and Junjie Hu. 2023. Retrieving

In Proc. of AAAI, pages 4541–4549. 

multimodal prompts for generative visual question

answering. In Findings of the ACL. 

Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. 

2022e. CLMLF:a contrastive learning and multi-

Sarah Partan and Peter Marler. 1999. Communication

layer fusion method for multimodal sentiment detec-

goes multimodal. Science, 283\(5406\):1272–1273. 

tion. In Findings of NAACL, pages 2282–2294. 

Soujanya Poria, Devamanyu Hazarika, Navonil Ma-

Sheng Liang, Mengjie Zhao, and Hinrich Schuetze. 

jumder, Gautam Naik, Erik Cambria, and Rada Mi-

2022. Modular and parameter-efficient multimodal

halcea. 2019a. MELD: A multimodal multi-party

fusion with prompting. In Findings of ACL, pages

dataset for emotion recognition in conversations. In 2976–2985. 

Proc. of ACL, pages 527–536. 

16061

Soujanya Poria, Navonil Majumder, Devamanyu Haz-Weizhou Shen, Siyue Wu, Yunyi Yang, and Xiaojun

arika, Deepanway Ghosal, Rishabh Bhardwaj, Sam-

Quan. 2021b. Directed acyclic graph network for

son Yu Bai Jian, Pengfei Hong, Romila Ghosh, Ab-

conversational emotion recognition. In Proc. of hinaba Roy, Niyati Chhaya, Alexander F. Gelbukh, 

ACL/IJCNLP, pages 1551–1560. 

and Rada Mihalcea. 2021. Recognizing emotion

cause in conversations. Cogn. Comput., 13\(5\):1317–

Dongming Sheng, Dong Wang, Ying Shen, Haitao

1332. 

Zheng, and Haozhuang Liu. 2020. Summarize before

aggregate: A global-to-local heterogeneous graph in-

Soujanya Poria, Navonil Majumder, Rada Mihalcea, 

ference network for conversational emotion recogni-

and Eduard H. Hovy. 2019b. Emotion recognition

tion. In Proc. of COLING, pages 4153–4163. 

in conversation: Research challenges, datasets, and

recent advances. IEEE Access, 7:100943–100953. 

Aman Shenoy, Ashish Sardana, and et al. 2020. 

Multilogue-net: A context-aware RNN for multi-

Tuomas Puoliväli, Satu Palva, and J. Matias Palva. 2020. 

modal emotion detection and sentiment analysis in

Influence of multiple hypothesis testing on repro-

conversation. In Workshop on Multimodal Language

ducibility in neuroimaging research: A simulation

\(Challenge-HML\), pages 19–28. 

study and python-based software. Journal of Neuro-science Methods, 337:108654. 

Apoorva Singh, Soumyodeep Dey, Anamitra Singha, 

and Sriparna Saha. 2022. Sentiment and emotion-

Preeth Raguraman, Mohan Ramasundaram, and Mid-

aware multi-modal complaint identification. In Proc. 

hula Vijayan. 2019. Librosa based assessment tool

of AAAI, pages 12163–12171. 

for music information retrieval systems. In Proc. of MIPR, pages 109–114. 

Tiening Sun, Zhong Qian, Sujun Dong, Peifeng Li, and

Qiaoming Zhu. 2022. Rumor detection on social

Wasifur Rahman, Md. Kamrul Hasan, Sangwu Lee, 

media with graph adversarial contrastive learning. In AmirAli Bagher Zadeh, Chengfeng Mao, Louis-Proc. of WWW, pages 2789–2797. 

Philippe Morency, and Mohammed E. Hoque. 2020. 

Integrating multimodal information in large pre-

Shiyin Tan, Jingyi You, and Dongyuan Li. 2022. 

trained transformers. In Proc. of ACL, pages 2359–

Temporality- and frequency-aware graph contrastive

2369. 

learning for temporal network. In Proc. of CIKM, pages 1878–1888. 

Vandana Rajan, Alessio Brutti, and Andrea Cavallaro. 

2022. Is cross-attention preferable to self-attention

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, 

for multi-modal emotion recognition? In Proc. of J. Zico Kolter, Louis-Philippe Morency, and Ruslan

ICASSP, pages 4693–4697. 

Salakhutdinov. 2019a. Multimodal transformer for

Nils Reimers and Iryna Gurevych. 2019. Sentence-

unaligned multimodal language sequences. In Proc. 

BERT: Sentence embeddings using Siamese BERT-

of ACL, pages 6558–6569. 

networks. In Proc. of EMNLP-IJCNLP, pages 3982–

3992. 

Yao-Hung Hubert Tsai, Paul Pu Liang, Amir Zadeh, 

Louis-Philippe Morency, and Ruslan Salakhutdinov. 

Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus

2019b. Learning factorized multimodal representa-

Hagenbuchner, and Gabriele Monfardini. 2009. The

tions. In proc. of ICLR. 

graph neural network model. IEEE Trans. Neural Networks, 20\(1\):61–80. 

Yao-Hung Hubert Tsai, Martin Ma, Muqiao Yang, 

Ruslan Salakhutdinov, and Louis-Philippe Morency. 

Brian B Schultz. 1985. Levene’s test for relative varia-

2020. Multimodal routing: Improving local and

tion. Systematic Zoology, 34\(4\):449–456. 

global interpretability of multimodal language analy-

sis. In Proc. of EMNLP, pages 1823–1833. 

Shiv Shankar. 2022. Multimodal fusion via cortical

network inspired losses. In Proc. of ACL, pages 1167–

Tana Wang, Yaqing Hou, Dongsheng Zhou, and Qiang

1178. 

Zhang. 2021. A contextual attention network for

multimodal emotion recognition in conversation. In Samuel Sanford Shapiro and Martin B Wilk. 1965. An

Proc. of IJCNN, pages 1–7. 

analysis of variance test for normality \(complete sam-

ples\). Biometrika, 52\(3/4\):591–611. 

Yan Wang, Jiayu Zhang, Jun Ma, Shaojun Wang, and

Jing Xiao. 2020. Contextualized emotion recognition

Noam Shazeer and Mitchell Stern. 2018. Adafactor:

in conversation as sequence tagging. In Proc. of

Adaptive learning rates with sublinear memory cost. 

SIGDIAL, pages 186–195. 

In Proc. of ICML, volume 80, pages 4603–4611. 

Yansen Wang, Ying Shen, Zhun Liu, Paul Pu Liang, 

Weizhou Shen, Junqing Chen, Xiaojun Quan, and Zhix-

Amir Zadeh, and Louis-Philippe Morency. 2019. 

ian Xie. 2021a. Dialogxl: All-in-one xlnet for multi-

Words can shift: Dynamically adjusting word rep-

party conversation emotion recognition. In Proc. of

resentations using nonverbal behaviors. In Proc. of AAAI, pages 13789–13797. 

AAAI, pages 7216–7223. 

16062

Yikai Wang, Xinghao Chen, Lele Cao, Wenbing Huang, Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cam-Fuchun Sun, and Yunhe Wang. 2022a. Multimodal

bria, and Louis-Philippe Morency. 2018. Multimodal

token fusion for vision transformers. In Proc. of

language analysis in the wild: CMU-MOSEI dataset

CVPR, pages 12176–12185. 

and interpretable dynamic fusion graph. In Proc. of ACL, pages 2236–2246. 

Yusong Wang, Dongyuan Li, Kotaro Funakoshi, and

Manabu Okumura. 2023. Emp: Emotion-guided

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-

multi-modal fusion and contrastive learning for per-

Philippe Morency. 2016. Multimodal sentiment in-

sonality traits recognition. In Proc. of ICMR, page

tensity analysis in videos: Facial gestures and verbal

243–252. 

messages. IEEE Intelligent Systems, 31\(6\):82–88. 

Zhen Wang. 2022. Modern question answering datasets

Jiaqi Zeng and Pengtao Xie. 2021. Contrastive self-

and benchmarks: A survey. CoRR, abs/2206.15030. 

supervised learning for graph classification. In Proc. 

of AAAI, pages 10824–10832. 

Zhen Wang, Xu Shan, Xiangxie Zhang, and Jie Yang. 

2022b. N24news: A new dataset for multimodal

Dong Zhang, Xincheng Ju, Wei Zhang, Junhui Li, 

news classification. In Proc. of LREC, pages 6768–

Shoushan Li, Qiaoming Zhu, and Guodong Zhou. 

6775. 

2021. Multi-modal multi-label emotion recognition

with heterogeneous hierarchical message passing. In Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, 

Proc. of AAAI, pages 14338–14346. 

Richang Hong, and Tat-Seng Chua. 2019. MMGCN:

multi-modal graph convolution network for person-

Yifei Zhang, Hao Zhu, Zixing Song, Piotr Koniusz, and

alized recommendation of micro-video. In Proc. of Irwin King. 2022. COSTA: covariance-preserving

MM, pages 1437–1445. 

feature augmentation for graph contrastive learning. 

In Proc. of KDD, pages 1–18. 

Yang Wu, Pengwei Zhan, Yunjian Zhang, LiMing Wang, 

and Zhen Xu. 2021. Multimodal fusion with co-

Ying Zhang, Hidetaka Kamigaito, and Manabu Oku-

attention networks for fake news detection. In Find-mura. 2023. Bidirectional transformer reranker for

ings of the ACL/IJCNLP, pages 2560–2569. 

grammatical error correction. In Findings of ACL, pages 3801–3825. 

Dongkuan Xu, Wei Cheng, Dongsheng Luo, Haifeng

Chen, and Xiang Zhang. 2021. Infogcl: Information-

Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu

aware graph contrastive learning. 

In Proc. of

Wu, and Liang Wang. 2020. Deep Graph Contrastive

NeurIPS, pages 30414–30425. 

Representation Learning. In ICML Workshop on Graph Representation Learning and Beyond. 

Jianing Yang, Yongxin Wang, Ruitao Yi, Yuying Zhu, 

Azaan Rehman, Amir Zadeh, Soujanya Poria, and

Louis-Philippe Morency. 2021. MTAG: modal-

temporal attention graph for unaligned human multi-

modal language sequences. In Proc. of NAACL-HLT, pages 1009–1021. 

Yiming Yang, Xin Liu, and et al. 1999. 

A re-

examination of text categorization methods. In Proc. 

of SIGIR, page 42–49. 

Jingyi You, Dongyuan Li, Manabu Okumura, and Kenji

Suzuki. 2022. JPG - jointly learn to align: Automated

disease prediction and radiology report generation. 

In Proc. of COLING, pages 5989–6001. 

Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, 

Zhangyang Wang, and Yang Shen. 2020. Graph con-

trastive learning with augmentations. In Proc. of NeurIPS, pages 1–12. 

Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu. 2021. 

Learning modality-specific representations with self-

supervised multi-task learning for multimodal senti-

ment analysis. In Proc. of AAAI, pages 10790–10797. 

Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cam-

bria, and Louis-Philippe Morency. 2017. Tensor fu-

sion network for multimodal sentiment analysis. In Proc. of EMNLP, pages 1103–1114. 

16063





A Example for Global Proximity

Symbols

Description

In Figure 7, given the network G and a modified xv ∈ R512

Video Features

xa ∈ R100

Audio Features

p, we first used the Katz index \(Katz, 1953\) to cal-xt ∈ R768

Text Features

culate a high-order similarity between the vertices. 

Contextual Representation Learning

We considered the arbitrary number of high-order

zg

v ∈ R512

Global Hidden Video Features

distances. For example, second-order similarity be-

zg

a ∈ R100

Global Hidden Audio Features

zg

t ∈ R768

Global Hidden Text Features

tween uA and

as

zg

1

uB

4

uA

1 → uB

4 = 0.83, third-order

m ∈ R1,380

Global Combined Features

z

similarity between

con ∈ R1,380

Topic-related Vector

uA and

as

1

uB

5

uA

1

→ uB5 =

ˆ

zg

m ∈ R1,380

Global Output Features

0.63, and fourth-order similarity between uA and

1

Specific Representation Learning

uB as

7

uA

1

→ uB7 = 0.21. We then define the

zℓv ∈ R460

Local Hidden Video Features

threshold score as 0.5, where a high-order similar-

zℓa ∈ R460

Local Hidden Audio Features

zℓ

ity score less than the threshold will not be selected

t ∈ R460

Local Hidden Text Features

bm ∈ R460

Basic Features

as added edges. Finally, we randomly selected p%

˜

zℓ\{v,a,t\} ∈ R460

Features in Shared Space

˜

b

edges \(whose scores are higher than the threshold

m ∈ R460

Basic Features in Shared Space

W\{v,a,t,b\} ∈ R460×460

Trainable Matrices

score\) and added them to the original graph G to

ˆ

zℓ\{v,a,t\} ∈ R460

New Multimodal Features

construct the augmented graph. 

ˆ

zℓm ∈ R1,380

New Multimodal Combined Features

zℓm ∈ R1380

Original Combined Features

Graph Contrastive Learning \(One GCN Layer\)

Speaker-A

\( ˆ

zℓ

\)

g ∥ ˆ

zℓm ∈ R2,760

Global-Local Combined Features

𝑢' 

AGG ∈ R2,760×2,760

Parameters of Aggregation Layer

\(

Speaker-B

COM ∈ R2,760×5,520

Input/Output of Combination Layer

W

𝑢" 

***High-order similarity Score***

graph ∈ R5,520×2,760

Dimention Reduction after COM

\! 

hm ∈ R2,760

Node Features of GCN Layer

𝑆\(𝑢" 

$

\! → 𝑢\# \) = 0.83

𝑢'\)

𝑆\(𝑢" 

$

\! → 𝑢% \) = 0.63

Table 8: Mathematical symbols for IEMOCAP dataset. 

" 

$

𝑢" 

𝑆\(𝑢& → 𝑢% \) = 0.51

$

***Threshold Scores = 0.5***

𝑆\(𝑢" → 𝑢$\) = 0.21

accuracy, judging from the averaged WF1 gain of

𝑢" 

' 

\! 

' 

%

𝑢& 

𝑆\(𝑢" 

$

the pair \(None, None\) in the upper left corners

& → 𝑢\# \) = 0.35

of Figure 8. In contrast, composing an original **Add p ratio edges. **

graph and its appropriate augmentation can benefit

𝑢"\#

the averaged WF1 of emotion recognition, judging

from the pairs \(None, any\) in the top rows or the

Figure 7: Example of adding p% high-order edges to

left-most columns of Figure 8. Similar observa-explore global topological information of graph. 

tion were in graphCL \(You et al., 2020\), without augmentation, GCL simply compares two original

B Dimensions of Mathematical Symbols

samples as a negative pair with the positive pair

loss becoming zero, which leads to homogeneously

Since we do not have much space to introduce

pushes all graph representations away from each

details about the dimensions of the mathematical

other. Appropriate augmentations can enforce the

symbols in our main body. We carefully list all

model to learn representations invariant to the de-

the dimensions of the mathematical symbols of

sired perturbations through maximizing the agree-

IEMOCAP in Table 8. Mathematical symbols for ment between a graph and its augmentation. 

other two datasets please see our source code. 

Obs.2: Composing different augmentations benefits the

C Observations of Graph Augmentation

model’s performance more. Applying augmentation

As shown in Figure 8, when we consider the combi-pairs of the same type does not often result in the

nations of \(FM

best performance \(see diagonals in Figure 8\). In

& EP\) and \(FP & GP\) as two graph

augmentation methods of the original graph, we

contrast, applying augmentation pairs of different

could achieve the best performance. Furthermore, 

types result in better performance gain \(see off-

we have the following observations:

diagonals of Figure 8\). Similar observations were in SimCSE \(Gao et al., 2021\). As mentioned in Obs.1: Graph augmentations are crucial. Without any

that study, composing augmentation pairs of dif-

data augmentation, GCL module will not improve

ferent types correspond to a “harder” contrastive

16064





**\(A\) IEMOCAP 4-emotion classification**

**\(B\) IEMOCAP 6-emotion classification**

**\(C\) MOSEI binary classification**

**\(D\) MOSEI multi-label classification**

Figure 8: Average WF1 gain when contrasting different augmentation pairs, compared with training without graph augmentation module. 

prediction task, which could enable learning more

D Parameters Sensitivity Study

generalizable representations. 

In this section, we give more details about param-

Obs.3: One view having two augmentations result in better eter sensitivity. First, as shown in Tables 9 & 10, 

performance. Generating each view by two aug-

when the window size ∈ \[6, 8\] for IEMOCAP \(6-

mentations further improve performance, i.e., the

way\) and the window size is 6 for IEMOCAP \(4-

augmentations FM & EP, or FM & GP. The aug-

way\), JOYFUL achieved the best performance. A

mentation pair \(FM & EP, FM & GP\) results in

small window size will miss much contextual in-

the largest performance gain compared with other

formation, and a large-scale window size contains

augmentation pairs. We conjectured the reason

too much noise \(topic will change over time\). We

is that simultaneously changing structural and at-

set the window size for past and future to 6. 

tribute information of the original graph can ob-

JOYFUL also has two hyper-parameters: α and

tain more heterogeneous contextual information

β, which balance the importance of MF module

for nodes, which can be consider as “harder” ex-

and GCL module in Eq.\(15\). Specifically, as shown ample to prompt the GCL model to obtain more

in Figure 9, we observed how α and β affect the generalizable and robust representations. 

performance of JOYFUL by varying α from 0.02 to

0.10 in 0.02 intervals and β from 0.1 to 0.5 in 0.1 in-

P&F

Happiness Sadness Neutral Anger Accuracy

WF1

tervals. The results indicate that JOYFUL achieved

size=1

83.27

83.04

80.63

81.54

81.87

81.82

the best performance when α ∈ \[0.06, 0.08\] and

size=2

79.02

82.92

83.93

86.65

83.46

83.41

β

size=3

80.88

86.34

84.07

85.64

84.52

84.45

∈ \[0.2, 0.3\] on IEMOCAP and and when α ∈

size=4

83.92

85.83

83.91

84.35

84.52

84.51

\[0.06, 0.1\] and β = 0.1 on MOSEI. The reason

size=5

82.93

87.85

83.79

86.47

85.26

85.20

size=6

81.73

86.42

85.17

88.46

85.58

85.56

why these parameters can affect the results is that

size=7

79.33

86.07

83.29

86.40

83.99

83.97

when

size=8

80.14

88.11

85.06

88.15

85.68

85.66

α< 0.06, MF becomes weaker and represen-

size=9

77.29

87.85

83.56

87.19

84.41

84.37

tations contain too much noise, which cannot pro-

size=10

80.00

87.47

85.29

88.64

85.68

85.66

size=ALL

79.87

84.35

83.20

84.75

83.24

83.24

vide a good initialization for downstream MERC

tasks. When α >0.1, it tends to make reconstruc-

Table 9: Results for various window sizes for graph

tion loss more important and JOYFUL tends to ex-

formation on the IEMOCAP \(4-way\). 

tract more common features among multiple modal-

ities and loses attention to explore features from

uni-modality. When β is small, graph contrastive

P&F

Hap. 

Sad. 

Neu. 

Ang. 

Exc. 

Fru. 

Acc. 

WF1

loss becomes weaker, which leads to indistinguish-

size=1

57.85 80.43 62.88 60.61 70.76 60.99 65.50 65.85

size=2

56.27 79.57 64.17 60.87 72.50 61.52 65.93 66.36

able representation. A larger β wakes the effect

size=3

60.80 80.26 66.06 64.47 73.17 62.70 67.71 68.09

of MF, leading to a local optimal solution. We set

size=4

59.95 80.79 67.96 67.18 71.60 64.89 68.64 69.05

size=5

60.06 81.42 68.23 66.33 73.88 63.24 68.76 69.17

α=0.06 and β=0.3 for IEMOCAP and MELD. We

size=6

60.94 84.42 68.24 69.95 73.54 67.55 70.55 71.03

size=7

59.84 80.53 67.93 68.12 73.72 63.91 68.82 69.26

set α=0.06 and β =0.1 for MOSEI. 

size=8

57.66 82.17 70.56 67.53 73.92 64.79 69.75 70.12

size=9

58.01 81.13 70.22 65.42 75.05 61.49 68.82 69.12

size=10

59.77 81.84 69.17 65.85 73.56 63.51 68.95 69.38

E Uni-modal Performance

size=ALL 54.74 78.75 66.58 64.56 68.63 63.46 66.42 66.80

The focus of this study was multimodal emo-

Table 10: Results for various window sizes for graph

tion recognition. However, we also compared

formation on the IEMOCAP \(6-way\). 

JOYFUL with uni-modal methods to evaluate its

performance of JOYFUL. We compared it with

16065





**\(A\) IEMOCAP 4-emotion classification \(B\) IEMOCAP 6-emotion classification** **\(C\) MOSEI binary classification**

**\(D\) MOSEI multi-label classification**

Figure 9: Parameters tuning for α and β on validation datasets for all multimodal emotion recognition tasks. 

Method

Modality

WF1

2019\), with 110 million parameters, as their text IEMOCAP 6-way

encoder without fine-tuning on ERC datasets. 

To verify whether the above inference is rea-

CESTa

Text

67.10

sonable, we used RoBERTa large model as our

SumAggGIN

Text

66.61

DiaCRN

Text

66.20

text feature extractor called Text \(RoBERTa-large\). 

DialogXL

Text

65.94

And we fine-tuned RoBERTa large model on the

DiaGCN

Text

64.18

COGMEN

Text

66.00

downstream IEMPCAP \(6-way\) dataset, following

DAG-ERC

Fine-tune Text \(RoBERTa-large\) 68.03

the same method of DAG-ERC called Fine-tune

Text \(Sentence-BERT\)

67.48

Text \(RoBERTA-large\). The observation meets our

JOYFUL

Text \(RoBERTa-large\)

68.05

intuition. With RoBERTa large model, JOYFUL im-

Fine-tune Text \(RoBERTa-large\) 68.45

proved the performance \(68.05 vs 67.48\) compared

A\+T\+V

71.03

with Sentence-BERT as our text encoder. And

Table 11: Overall performance comparison on MOSEI

JOYFUL could obtain better performance \(68.45

with Text Modality. 

vs 68.03\) in terms of WF1 than DAG-ERC with

fine-tuned RoBERTa-large, demonstrating that fine-

tuning large-scale model can help obtain richer text

DAG-ERC \(Shen et al., 2021b\), CESTa \(Wang

features to improve the performance. However, 

et al., 2020\), SumAggGIN \(Sheng et al., 2020\), 

considering a fair comparison with other multi-

DiaCRN \(Hu et al., 2021\), DialogXL \(Shen et al., 

modal emotion recognition baselines \(they do not

2021a\), DiaGCN \(Ghosal et al., 2019\), and COG-have the fine-tuning process \(Joshi et al., 2022; 

MEN \(Joshi et al., 2022\). Following COGMEN, 

Ghosal et al., 2019\)\) and saving the additional text-based models were specifically optimized for

time-consuming on fine-tuning, we directly adopt

text modalities and incorporated changes to ar-

Sentence-BERT as our text encoder for IEMOCAP. 

chitectures to cater to text. As shown in Ta-

ble 11, J

F Pseudo-Code of J

OYFUL, being a fairly generic architecture, 

OYFUL

still achieved better or comparable performance

As shown in Algorithm 1, to make JOYFUL easy to with respect to the state-of-the-art uni-modal meth-understand, we also provide a pseudo-code. 

ods. Adding more information via other modali-

ties helped to further improve the performance of

G Benjamini-Hochberg Correction

JOYFUL \(Text vs A\+T\+V\). When using only text

modality, the DAG-ERC baseline could achieve

Benjamini-Hochberg Correction \(B-H\) \(Benjamini

higher WF1 than J

and Hochberg, 1995\) is a powerful tool that de-OYFUL. And we conjecture the

main reasons is: DAG-ERC \(Shen et al., 2021b\)

creases the false discovery rate. Considering the

fine-tuned RoBERTa large model \(Liu et al., 2019\), 

reproducibility of the multiple significant test, we

with 354 million parameters, as their text encoder. 

introduce how we adopt the B-H correction and

By fine-tuning on RoBERTa large model under

give the hyper-parameter values that we used. We

the guidance of downstream emotion recognition

first conduct a t-test \(Yang et al., 1999\) with default signals, RoBERTa large model can provide the

parameters2 to calculate the p-value between each most suitable text features for ERC. Compared

comparison method with JOYFUL. We then put the

with DAG-ERC, J

individual p-values in ascending order as input to

OYFUL and other methods di-

rectly use Sentence-BERT \(Reimers and Gurevych, 

2scipy.stats.ttest\_ind.html

16066

Algorithm 1: Overall process of JOYFUL

were greater than 0.05. This indicates that the re-

input :Visual features x

sults of the baselines and our model all adhere

v ; 

Audio features xa; 

to the assumption of normality. For example, in

Text features xt; 

IEMOCAP-4, p-values for \[Mult, RAVEN, MTAG, 

Parameters: α, β, Window size

output :Emotion recognition label. 

PMR, MICA, COGMEN, JOYFUL\] are \[0.903, 

Initialize trainable parameters; 

0.957, 0.858, 0.978, 0.970, 0.969, 0.862\]. Further-

for epoch ← 1 to epoch num do

more, we used the Levene’s test \(Schultz, 1985\)

Global Contextual Fusion ˆ

zgm; 

to check for homogeneity of variances between

Specific Modality Fusion ˆ

zℓm=\(zgv∥zga∥zg\); 

t

//

baselines and our model. Under the constraint of

**Compute multimodal fusion loss**

Compute Lmf, in accordance with Eq.\(8\); 

a significance level \(alpha = 0.05\), we found that

Feature Concatenation h = \(ˆ

zgm∥ ˆ

zℓm\); 

our p-values are greater than 0.05, indicating the

Adopt h as initialization for Graph; 

homogeneity of the variances between the base-

// **Generate two augmented views**

Apply FM & EP to generate view: G\(1\); 

lines and our model. For example, we obtained

Apply FM & GP to generate view: G\(2\); 

p-values 0.3101 and 0.3848 for group-based base-

// **Extract features of two views**

lines on IEMOCAP-4 and IEMOCAP-6, respec-

H\(1\) = GCN s\(G\(1\)\), H\(2\) = GCNs\(G\(2\)\) ; 

tively. Since we were able to demonstrate that all

// **Compute contrastive learning loss**

Compute L

baselines and our model conform to the assump-

ct, in accordance with Eq.\(13\) ; 

// **Aggregate extracted features**

tions of normality and homogeneity of variances, 

H = H\(1\) \+ H\(2\) ; 

we believe that the significance tests we reported

// **Compute emotion recognition loss**

Compute

are accurate. 

Lce, in accordance with Eq.\(14\); 

// **Joint training**

Compute L

H Representation Visualization

all, in accordance with Eq.\(15\); 

// **Optimize with Adam optimizer**

Adopt classifier on

We visualized the node features to understand the

H to predict the emotional label. 

function of the multimodal fusion mechanism and

the GCL-based node representation learning com-

calculate the p-value corrected using the B-H cor-

ponent, as shown in Figure 10. Figure 10 \(A\) shows rection. We directly use the “multipletests\(\*args\)” 

the concatenated multimodal features on the input

function from python package3 and set the hyper-side. Figure 10 \(B\) shows the representation of parameter of the false discovery rate Q = 0.05, 

utterances after the feature fusion module. Fig-

which is a widely used default value \(Puoliväli

ure 10 \(C\) shows the representation of the utter-

et al., 2020\). Finally, we obtain a cut-off value ances after the GCL module \(Eq.\(10\)\) and before

as the output of the multipletests function, where

the pre-softmax layer \(Eq.\(11\)\). We observed that

cut-off is a dividing line that distinguishes whether

utterances could be roughly separated after the fea-

two groups of data are significant. If the p-value

ture fusion mechanism, which indicates that the

is smaller than the cut-off value, we can conclude

multimodal fusion mechanism can learn distinctive

that two groups of data are significantly different. 

features to a certain extent. After GCL-based mod-

The use of t-test for testing statistical signifi-

ule, JOYFUL can be easily separated, demonstrating

cance may not be appropriate for F-scores, as men-

that GCL can provide distinguishable representa-

tioned in Dror et al. \(2018\), as we cannot assume tion by exploring vertex attributes, graph structure, 

normality. To verify whether our data meet the

and contextual information from datasets. 

normality assumption and the homogeneity of vari-

ances required for the t-test, following Shapiro and

I Labels Distribution of Datasets

Wilk \(1965\) and Levene et al. \(1960\), we conducted In this section, we list the detailed label distribu-the following validation. First, we performed the

tion of the three multimodal emotion recognition

Shapiro-Wilk test on each group of experimental

datasets MELD \(Table 12\), IEMOCAP 4-way \(Ta-results to determine whether they are normally dis-

ble 13\), IEMOCAP 6-way \(Table 14\) and MOSEI tributed. Under the constraint of a significance

\(Table 15\) in the draft. 

level \(alpha=0.05\), all p-values resulting from the

Shapiro-Wilk test 4 for the baselines and our model J Multimodal Sentiment Analysis

3statsmodels.stats.multitest.multipletests.html

We conducted experiments on two publicly avail-

4scipy.stats.shapiro.html

able datasets, MOSI \(Zadeh et al., 2016\) and MO-16067





**\(A\) Initial feature visualization**

**\(B\) Feature fusion visualization**

**\(C\) Feature GCL visualization**

Figure 10: t-SNE visualization of IEMOCAP \(6-way\) features. 

MELD

Train

Valid

Test

IEMOCAP 6-way

Train

Valid

Test

Anger

1,109

153

345

Happy

459

45

144

Disgusted

271

22

68

Sad

746

93

245

Fear

268

40

50

Neutral

1,161

163

384

Joy

1,743

163

402

Angry

854

79

170

Neutral

4,710

470

1,256

Excited

576

166

299

Sadness

683

111

208

Frustrated

1,350

118

381

Surprise

1,205

150

281

Total

5,146

644

1,623

Total

9,989

1,109

2,610

Table 14: Labels distribution of IEMOCAP 6-way. 

Table 12: Labels distribution of MELD dataset. 

MOSEI

Train

Valid

Test

IEMOCAP 4-way

Train

Valid

Test

Happy

8,735

1,005

2,505

Happy

453

51

144

Sad

4,269

520

1,129

Sad

783

56

245

Angry

3,526

338

1,071

Neutral

1,092

232

384

Surprise

1,642

203

441

Angry

872

61

170

Disgusted

2,955

281

805

Fear

1,331

176

385

Total

3,200

400

943

Total

22,458

2,523

6,336

Table 13: Labels distribution of IEMOCAP 4-way. 

Table 15: Labels distribution of MOSEI dataset. 

SEI \(Zadeh et al., 2018\), to investigate the perfor-intervals between -3 and \+3 as the corresponding

mance of JOYFUL on the multimodal sentiment

truths. And binary classification accuracy \(ACC-2\)

analysis \(MSA\) task. 

was computed for non-negative/negative classifica-

\) Datasets: MOSI contains 2,199 utterance video

tion results. 

segments, and each segment is manually annotated

\) Baselines: We compared JOYFUL with three

with a sentiment score ranging from -3 to \+3 to

types of advanced multimodal fusion frameworks

indicate the sentiment polarity and relative senti-

for the MSA task as follows, including current

ment strength of the segment. MOSEI contains

SOTA baselines MMIM \(Han et al., 2021b\) and 22,856 movie review clips from the YouTube web-BBFN \(Han et al., 2021a\): \(1\) Early multimodal fu-site. Each clip is annotated with a sentiment score

sion methods, which combine the different modal-

and an emotion label. And the exact number of sam-

ities before they are processed by any neural

ples for training/validation/test are 1,284/229/686

network models. We utilized Multimodal Fac-

for MOSI and 16,326/1,871/4,659 for MOSEI. 

torization Model \(MFM\) \(Tsai et al., 2019b\), 

\) Metrics: Following previous studies \(Han et al., 

and Multimodal Adaptation Gate BERT \(MAG-

2021a; Yu et al., 2021\), we utilized evaluation met-BERT\) \(Rahman et al., 2020\) as baselines. \(2\) rics: mean absolute error \(MAE\) measures the ab-Late multimodal fusion methods, which combine

solute error between predicted and true values. Per-

the different modalities before the final decision

son correlation \(Corr\) measures the degree of pre-

or prediction layer. We utilized multimodal Trans-

diction skew. Seven-class classification accuracy

former \(MuIT\) \(Tsai et al., 2019a\), and modal-

\(ACC-7\) indicates the proportion of predictions

temporal attention graph \(MTAG\) \(Yang et al., 

that correctly fall into the same interval of seven

2021\) as baselines. \(3\) Hybrid multimodal fusion 16068

Case

Input modality

Target

Text

Visual

Acoustic

MSA

MERC

Case A Plot to it than that the action scenes were Smiling face Stress

\+1.666

Positive

my favorite parts through it’s. 

Relaxed wink

Pitch variation

Case B You must promise me that you’ll survive, Full of tears The voice is

-1.200 Negative

you won’t give up. 

in his eyes

weak and trembling

Table 16: Case study on the importance of each modality for MSA and MERC tasks. Blue in Text modality marks the contents including the strength of sentiments. Underline marks fragments contributing to the target on MERC. 

methods combine early and late multimodal fu-

Method

MOSI

MOSEI

sion mechanisms to capture the consistency and the

MAE ↓Corr ↑Acc-7 ↑Acc-2 ↑ MAE ↓Corr ↑Acc-7 ↑Acc-2 ↑

difference between different modalities simultane-

MFM

0.877 0.706 35.4

81.7

0.568 0.717 51.3

84.4

ously. We utilized modality-invariant and modality-

MAG-BERT

0.731 0.789

✗

84.3

0.539 0.753

✗

85.2

MulT

0.861 0.711

✗

84.1

0.580 0.703

✗

82.5

specific representations for MSA \(MISA\) \(Haz-

MTAG

0.866 0.722 0.389

82.3

✗

✗

✗

✗

MISA

0.804 0.764

✗

82.10

0.568 0.724

✗

84.2

arika et al., 2020\), Self-Supervised multi-task learn-Self-MM

0.713 0.789

✗

85.98

0.530 0.765

✗

85.17

ing for MSA \(Self-MM\) \(Yu et al., 2021\), Bi-BBFN

0.776 0.755 45.00 84.30

0.529 0.767 54.80 86.20

Bimodal Fusion Network \(BBFN\) \(Han et al., 

MMIM

0.700 0.800 46.65 86.06 0.526 0.772 54.24 85.97

2021a\), and MultiModal InfoMax \(MMIM\) \(Han

JOYFUL\+MAE 0.711 0.792 45.58 85.87

0.529 0.768 53.94 85.68

et al., 2021b\) as baselines. 

Table 17: Experimental results on the MOSI and MO-

\) Implementation Details: The results of pro-

posed J

SEI datasets. ✗ indicates unreported results. Bold in-

OYFUL were averaged over ten runs using

dicates the least MAE, highest Corr, Acc-7, and Acc-2

random seeds. We keep all hyper-parameters and

scores for each dataset. 

implementations the same as in the MERC task

reported in Sections 4.1 and 4.2. To make JOYFUL

fit in the MSA task, we replace the current cross-

more attention to the text modality than visual and

entropy loss L

acoustic modalities during multimodal feature fu-

ce in Eq. \(15\) by mean absolute error

loss L

sion, they may achieve low MAE, high Corr, Acc-2, 

mae as follows:

and Acc-7. Specifically, BBFN \(Han et al., 2021a\)

m

1 X

proposed a Bi-bimodal fusion network to enhance

Lmae =

|ˆy

m

i − yi|, 

\(16\)

the text modality’s importance by only considered

i=1

text-visual and text-acoustic interaction for fea-

where ˆyi is the predicted value for the i-th sample, 

tures fusion. Conversely, considering the three

yi is the truth label for the i-th label, m is the total

modalities are all important for the MERC task

number of samples, and | · | is the L1 norm. We

as presented in Table 16, we designed JOYFUL to denote this model as JOYFUL\+MAE. 

utilize the concatenation of the three modalities

Experimental results on the MOSI and MOSEI

representations for prediction. Similar to our pro-

datasets are listed in Table 17. Although the proposal, MISA and MAG-BERT considered the three

posed JOYFUL could outperform most of the base-

modalities equally important during feature fusion

lines \(above the blue line\), it performs worse than

but performed worse than SOTA baselines on the

current SOTA models: BBFN and MMIM \(below

MSA task. In our consideration, because of such at-

the blue line\). We conjecture the main reasons

tention to modalities, JOYFUL outperformed SOTA

are: when determining the strength of sentiments, 

baselines on the MERC task but underperformed

compared with visual and acoustic modalities that

SOTA baselines on the MSA task. 

may contain much noise data, text modality is more

important for prediction \(Han et al., 2021a\). Table 16 lists such examples, where textual modality is more indicative than other modalities for the

MSA task. Because the two baselines: BBFN \(Han

et al., 2021a\) and MMIN \(Han et al., 2021b\), pay 16069



