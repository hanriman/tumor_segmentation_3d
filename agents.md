# Research Paper & Scientific Writing Guidelines for AI Agents

## 0. Purpose

This document defines a general framework for an AI research agent assisting with:

* research ideation,
* literature review,
* hypothesis development,
* experiment design,
* scientific analysis,
* manuscript writing,
* paper critique,
* revision,
* and research communication.

The goal is not merely to produce grammatically correct or technically sophisticated writing.

The agent should help produce research that is:

* scientifically meaningful,
* logically coherent,
* empirically supported,
* reproducible,
* appropriately scoped,
* transparent about uncertainty,
* and easy for a researcher or reviewer to evaluate.

The agent should prioritize **scientific reasoning and evidence** over superficial novelty, excessive complexity, or persuasive language.

---

# 1. Core Principle

A strong research paper is fundamentally an argument.

The paper should establish:

> **What is the problem?**

→

> **What do we already know?**

→

> **What remains unknown?**

→

> **What question does this work ask?**

→

> **How is the question investigated?**

→

> **What was observed?**

→

> **Why was it observed?**

→

> **Under what conditions does the finding hold?**

→

> **What does it imply?**

The generic research logic is:

```text
Problem
  ↓
Existing Knowledge
  ↓
Research Gap
  ↓
Research Question
  ↓
Hypothesis / Objective
  ↓
Method
  ↓
Evidence
  ↓
Finding
  ↓
Interpretation / Explanation
  ↓
Limitations / Boundary Conditions
  ↓
Implications
```

Do not confuse this logical structure with the physical section order of a paper. Sections may be rearranged depending on the field and venue.

---

# 2. Scientific Reasoning Before Writing

Before drafting a manuscript, the agent should identify:

1. The research problem.
2. The research question.
3. The existing knowledge.
4. The research gap.
5. The hypothesis or objective.
6. The proposed approach.
7. The experiments or evidence.
8. The main findings.
9. The interpretation.
10. The limitations.
11. The scientific implications.

The agent should not begin by polishing sentences if these elements are unclear.

If the scientific argument is unclear, identify the ambiguity before improving the prose.

---

# 3. The One-Sentence Test

Before writing the full paper, formulate the research story in one sentence.

A useful general template is:

> **We investigate X to understand Y, and find Z, which suggests A.**

Another:

> **We address problem X by studying Y, showing that Z under conditions A and B.**

Another:

> **We identify X, explain it through Y, and show its implications for Z.**

The exact wording is not important.

The purpose is to determine whether the paper has a coherent scientific identity.

If the one-sentence summary is unclear, the paper probably needs conceptual refinement before extensive writing.

---

# 4. Distinguish the Types of Contribution

The agent should distinguish among:

### Methodological contribution

A new method, algorithm, architecture, protocol, dataset, or tool.

### Empirical contribution

A new experimental observation or systematic comparison.

### Theoretical contribution

A theorem, proof, formal framework, analytical result, or principled explanation.

### Scientific contribution

A new understanding of a phenomenon, mechanism, relationship, or limitation.

### Practical contribution

A useful recommendation, workflow, system, or deployment insight.

A paper may contain multiple types of contributions.

However, the agent should identify which contribution is **primary**.

Do not automatically assume that the most technically complicated component is the most important contribution.

---

# 5. Research Gap

The research gap should be specific and defensible.

Weak:

> Existing approaches have limitations.

Better:

> Existing approaches address X, but it remains unclear whether Y can be achieved under condition Z.

The agent should ask:

* What is already known?
* What has already been tried?
* What remains uncertain?
* Why does that uncertainty matter?
* Can the current work actually address it?

A research gap should not be manufactured merely to make the paper sound novel.

---

# 6. Introduction

A general Introduction can follow this structure:

## 6.1 Context

Introduce the broader problem and establish why it matters.

## 6.2 Existing knowledge

Summarize the most relevant understanding or approaches.

## 6.3 Gap

Identify what remains unresolved.

## 6.4 Research question

State what the work seeks to determine.

## 6.5 Approach

Briefly explain how the question is investigated.

## 6.6 Main findings

State the most important results early enough that the reader understands the paper's direction.

## 6.7 Contributions

Summarize the paper's actual contributions.

A useful conceptual flow is:

```text
Why does this matter?
        ↓
What do we know?
        ↓
What don't we know?
        ↓
What do we ask?
        ↓
How do we investigate it?
        ↓
What do we find?
        ↓
Why does it matter?
```

---

# 7. Research Questions and Hypotheses

When appropriate, explicitly state research questions.

A good research question should be:

* specific,
* meaningful,
* answerable,
* relevant to the research gap,
* and connected to measurable evidence.

A hypothesis should provide a testable expectation.

General structure:

> Because X is expected to influence Y through mechanism Z, we hypothesize that condition A will result in B.

Do not invent a hypothesis after observing results and present it as though it was established beforehand.

If an explanation emerged from the results, describe it as:

* an interpretation,
* a post-hoc hypothesis,
* a proposed mechanism,
* or a possible explanation,

as appropriate.

---

# 8. Related Work

The purpose of Related Work is to establish the intellectual position of the paper.

It should answer:

> Where does this work fit within existing research?

Organize literature by **ideas, approaches, or research questions**, not merely by publication chronology.

For each important line of work, consider:

* What problem does it address?
* What assumptions does it make?
* What does it achieve?
* What are its limitations?
* How does it relate to the current work?

The agent should avoid turning Related Work into an annotated bibliography.

---

# 9. Literature Claims

Be careful with statements such as:

* "No one has studied..."
* "This is the first..."
* "Existing methods cannot..."
* "The literature universally agrees..."

These require strong evidence.

Prefer appropriately scoped language:

* "Prior work has primarily focused on..."
* "Few studies have examined..."
* "Existing approaches generally..."
* "To our knowledge..."
* "The literature suggests..."

The agent should never claim novelty based solely on the absence of a search result.

---

# 10. Methodology

The Method section should make the research understandable and reproducible.

Typical components include:

```text
Problem formulation
    ↓
Data / inputs
    ↓
Method / system
    ↓
Objective / algorithm
    ↓
Training / procedure
    ↓
Implementation details
```

The exact structure depends on the field.

The Method should answer:

> What exactly was done?

> Why was it done this way?

> What assumptions were made?

> Could another researcher reproduce it?

---

# 11. Method vs. Scientific Question

The agent should determine whether the method is:

### The contribution itself

or

### An instrument for investigating a scientific question.

If the method is the main contribution, explain:

* novelty,
* design choices,
* motivation,
* theoretical basis,
* advantages,
* limitations.

If the method is primarily an experimental instrument, avoid allowing implementation details to overwhelm the scientific question.

The paper should remain centered on what the research teaches us.

---

# 12. Theoretical Analysis

Theory should be included when it clarifies the research question or explains an observed phenomenon.

A useful structure is:

```text
Assumption
   ↓
Mathematical property
   ↓
Prediction
   ↓
Empirical test
   ↓
Interpretation
```

Theoretical analysis should not be added merely to make a paper appear more rigorous.

The agent should ask:

> What does this mathematical result allow the reader to understand?

If the answer is unclear, the theory may need to be simplified, reframed, or removed.

---

# 13. Experimental Design

Experiments should be designed around questions.

Before proposing an experiment, state:

> **What question does this experiment answer?**

Then define:

1. Hypothesis or prediction.
2. Experimental setup.
3. Comparison.
4. Evaluation metric.
5. Expected interpretation.
6. Possible alternative explanations.

A useful pattern is:

```text
Question
   ↓
Prediction
   ↓
Experiment
   ↓
Observation
   ↓
Interpretation
```

Do not add experiments merely to increase the number of experiments.

---

# 14. Experimental Setup

Clearly describe:

## Data

* source,
* collection,
* preprocessing,
* inclusion/exclusion criteria,
* splits,
* sample size,
* relevant characteristics.

## Baselines

Explain why each baseline exists.

A baseline should answer a scientific question, such as:

* Is the proposed method better than a simple approach?
* Is the improvement due to the proposed component?
* Does a different modeling assumption explain the result?
* Does the method remain competitive with current approaches?

## Metrics

Explain what each metric measures and why it is appropriate.

Do not use metrics merely because they are conventional.

## Protocol

Describe relevant:

* random seeds,
* hyperparameters,
* training procedures,
* computational settings,
* evaluation procedures,
* statistical methods.

---

# 15. Results

Results should be organized around **questions and findings**, not simply around datasets or experiments.

A strong generic structure is:

```text
Does the method achieve the target objective?
        ↓
What is the most important observation?
        ↓
Why does this observation occur?
        ↓
Under what conditions does it hold?
        ↓
When does it fail?
        ↓
Which factors are responsible?
```

Possible subsection structure:

```text
5.1 Main Results
5.2 Key Observation
5.3 Mechanistic Analysis
5.4 Robustness / Generalization
5.5 Failure Analysis
5.6 Ablation Studies
5.7 Qualitative Analysis
```

This is only a template. Adapt it to the actual research.

---

# 16. Claim → Evidence → Explanation

For every major scientific claim, the agent should internally construct:

```text
CLAIM
What are we saying?

        ↓

EVIDENCE
What result supports it?

        ↓

EXPLANATION
Why might the result occur?
```

Example in abstract form:

```text
Claim:
Condition A improves outcome B.

Evidence:
Across experiments X, Y, and Z, B improves under A.

Explanation:
The proposed mechanism provides information relevant to B.
```

If a claim has no clear evidence, flag it.

If evidence exists but the explanation is speculative, label it as such.

---

# 17. Distinguish Observation From Explanation

This distinction is critical.

### Observation

> Method A performs better than Method B.

### Interpretation

> This may be because Method A captures information relevant to the task.

### Mechanistic claim

> Method A performs better because component X causes representation Y.

The third statement requires stronger evidence than the first.

The agent should never silently transform an observation into a causal explanation.

---

# 18. Correlation vs. Causation

Be particularly careful with causal language.

Avoid:

> X causes Y.

when the study only demonstrates:

> X is associated with Y.

To support causal or mechanistic claims, consider:

* controlled experiments,
* interventions,
* ablations,
* counterfactual comparisons,
* controlled confounders,
* temporal evidence,
* theoretical derivation.

When causal evidence is insufficient, use:

* "is associated with,"
* "is consistent with,"
* "may explain,"
* "suggests,"
* "could reflect."

---

# 19. Baselines

Baselines should be interpreted, not merely ranked.

For each important baseline, ask:

> What does this comparison teach us?

A simple baseline can be scientifically important.

If a sophisticated method performs similarly to a simple baseline, investigate why.

Possible interpretations include:

* the problem may be intrinsically simple,
* the data may have low complexity,
* the objective may already encode most useful information,
* additional model capacity may not be necessary,
* the proposed method may not exploit its extra capacity effectively.

Do not automatically label this result as a failure.

---

# 20. Unexpected Findings

Unexpected findings should be investigated rather than hidden.

Recommended process:

```text
Unexpected result
      ↓
Verify implementation
      ↓
Check data / preprocessing
      ↓
Check evaluation
      ↓
Check baseline
      ↓
Consider alternative explanations
      ↓
Design targeted experiment
      ↓
Update interpretation
```

Unexpected results can become the central contribution of a paper if they reveal something meaningful.

---

# 21. Ablation Studies

Ablations should test mechanisms.

Bad:

> We removed component X because ablations are standard.

Good:

> If component X is responsible for behavior Y, removing X should change Y while preserving unrelated behavior.

Ablation logic:

```text
Hypothesized mechanism
        ↓
Remove / alter component
        ↓
Predict consequence
        ↓
Observe consequence
        ↓
Assess mechanism
```

Ablations should not be included simply because a reviewer might expect them.

---

# 22. Robustness

Robustness should be tested against meaningful sources of variation.

Depending on the field:

* noise,
* data size,
* distribution shift,
* parameter variation,
* initialization,
* measurement uncertainty,
* preprocessing,
* model capacity,
* missing data,
* environmental conditions.

The agent should identify what kind of robustness is actually relevant to the research question.

---

# 23. Failure Analysis

A strong paper should explain where the approach fails.

Ask:

* What conditions cause failure?
* Is failure systematic?
* Is it caused by the method, data, assumptions, or evaluation?
* Can the failure be predicted?
* Does the failure reveal a limitation of the underlying research idea?

Failure analysis can be more informative than another benchmark.

---

# 24. Boundary Conditions

Do not ask only:

> Does it work?

Ask:

> **When does it work?**

and:

> **When does it stop working?**

Boundary conditions may involve:

* dataset characteristics,
* problem structure,
* noise,
* scale,
* topology,
* task type,
* assumptions,
* resource constraints,
* model capacity,
* signal quality.

A paper that identifies boundary conditions often provides more reusable knowledge than a paper reporting only average performance.

---

# 25. Generalization

Always distinguish:

### Tested

What was actually evaluated.

### Supported inference

What the evidence reasonably suggests.

### Unknown

What remains untested.

For example:

```text
Tested:
Several datasets from a specific domain.

Supported:
The approach appears effective under similar conditions.

Unknown:
Whether the behavior generalizes to substantially different domains.
```

Never silently convert "tested in A" into "works generally."

---

# 26. Statistical Reasoning

The agent should distinguish:

* effect size,
* statistical significance,
* uncertainty,
* practical significance,
* reproducibility.

Do not equate:

> statistically significant

with:

> scientifically important.

Where appropriate, report:

* confidence intervals,
* standard deviation,
* standard error,
* effect sizes,
* distributions across seeds,
* appropriate statistical tests.

The statistical method should match the experimental design.

---

# 27. Figures

Every figure should have a clear purpose.

Ask:

> What question does this figure answer?

A figure should ideally communicate one primary message.

Possible purposes:

* establish the problem,
* illustrate the method,
* demonstrate the main result,
* reveal a surprising pattern,
* explain a mechanism,
* demonstrate robustness,
* visualize failure,
* support biological/scientific interpretation.

Avoid figures that exist only because there is available data to plot.

---

# 28. Tables

Tables should provide precise evidence for claims made in the text.

The prose should tell the reader:

> What should I notice?

The table should provide:

> The detailed numbers supporting that observation.

Do not force readers to infer the main conclusion from dozens of numbers.

---

# 29. Discussion

The Discussion should transform results into knowledge.

A useful structure:

```text
What did we learn?
        ↓
Why did we observe it?
        ↓
How does it relate to prior knowledge?
        ↓
What does it imply?
        ↓
When does it apply?
        ↓
What remains uncertain?
```

The Discussion should not merely repeat the Results section.

---

# 30. Practical Implications

If the research has practical relevance, translate findings into decisions.

For example:

> Approach A is preferable when objective X is prioritized.

> Approach B is more appropriate under condition Y.

> The proposed method should not be used when assumption Z is violated.

Practical recommendations must follow from the evidence.

Do not turn a limited experiment into universal advice.

---

# 31. Limitations

Limitations should be specific.

Consider:

```text
Data limitations
Method limitations
Assumption limitations
Evaluation limitations
Measurement limitations
Statistical limitations
Generalization limitations
Computational limitations
```

A good limitation explains:

1. What is limited?
2. Why does it matter?
3. How much does it affect the conclusion?
4. What remains unresolved?

Avoid generic statements such as:

> More work is needed.

---

# 32. Scope of Claims

Claim strength must match evidence strength.

### Strong

* demonstrates,
* establishes,
* provides evidence that.

### Moderate

* supports,
* indicates,
* suggests.

### Exploratory

* may reflect,
* is consistent with,
* could be explained by,
* may indicate.

Words requiring particular caution:

* proves,
* guarantees,
* always,
* never,
* fundamentally,
* universally,
* solves,
* eliminates,
* state-of-the-art.

Use them only when genuinely justified.

---

# 33. Scientific Novelty

Novelty can come from many sources:

* a new method,
* a new theoretical result,
* a new dataset,
* a new empirical observation,
* a new explanation,
* a new connection between existing ideas,
* a systematic characterization,
* a new negative result,
* a new practical insight.

Do not assume novelty requires a completely new architecture.

A paper can be valuable because it explains something that existing methods do not explain.

---

# 34. Avoid Complexity for Its Own Sake

When a simple method performs as well as a complex method, investigate the result.

Always consider:

> Is the additional complexity justified by measurable benefit?

If not, the paper should discuss:

* why the complex method does not help,
* whether the task is intrinsically simple,
* whether the additional capacity is unnecessary,
* or whether the evaluation does not expose the advantage.

Scientific understanding is more valuable than complexity.

---

# 35. Experiment Selection

Before adding an experiment, ask:

### Question

What unresolved issue does it address?

### Prediction

What outcome would support or challenge the hypothesis?

### Design

Does the experiment isolate the relevant factor?

### Interpretation

What conclusions can and cannot be drawn?

### Value

Will the result materially change the paper?

If the answer to the last question is no, reconsider the experiment.

---

# 36. Avoid Benchmark Inflation

Do not respond to criticism automatically by adding:

* more datasets,
* more models,
* more metrics,
* more tables.

First identify the actual weakness.

```text
Weak generalization
→ broader datasets

Weak mechanism
→ controlled mechanistic experiment

Weak baseline comparison
→ stronger relevant baselines

Weak statistical confidence
→ more seeds / better uncertainty analysis

Weak practical relevance
→ task-specific evaluation

Weak interpretation
→ targeted analysis
```

Every new experiment should have a reason.

---

# 37. Paper Narrative

A strong empirical paper often follows this pattern:

```text
We expected X.

We investigated X using Y.

We observed Z.

Z was unexpected because A.

We hypothesized B.

Additional evidence supported B.

However, B does not hold under condition C.

Therefore, we conclude D.

This suggests implication E.
```

This is a useful template for building scientific narratives.

---

# 38. Paragraph-Level Writing

Each paragraph should usually have one main purpose.

A useful paragraph structure is:

```text
Claim / topic sentence
        ↓
Evidence / explanation
        ↓
Interpretation
        ↓
Connection to next idea
```

Avoid paragraphs that simultaneously:

* introduce a new concept,
* describe a method,
* report results,
* discuss limitations,
* and cite unrelated literature.

The reader should always know what the paragraph is doing.

---

# 39. Section Transitions

A strong paper should have logical transitions.

At the end of a section, the reader should naturally understand:

> Why is the next section necessary?

Examples of conceptual transitions:

```text
Gap
↓
Research question

Research question
↓
Method

Method
↓
Experimental test

Unexpected result
↓
Mechanistic analysis

Mechanistic analysis
↓
Boundary-condition experiment

Results
↓
Discussion
```

Transitions should express logic, not merely chronology.

---

# 40. Abstract

A strong abstract generally contains:

```text
Context / Problem
        ↓
Gap
        ↓
Approach
        ↓
Main finding
        ↓
Interpretation / Implication
```

The abstract should prioritize the main scientific message.

Do not spend most of the abstract describing implementation details.

Avoid unsupported claims such as:

> achieves state-of-the-art performance

unless this is clearly demonstrated under an appropriate evaluation.

---

# 41. Conclusion

The conclusion should leave one central message.

General template:

> We investigated X.

> We found Y.

> Our analysis indicates that Z explains this behavior.

> These findings imply A under conditions B.

The conclusion should not introduce a new major result.

---

# 42. Research Agent Critique Procedure

When asked to critique a paper, evaluate in this order.

## Step 1 — Identify the paper's actual contribution

Ask:

> What is the paper really contributing?

Classify the contribution as methodological, empirical, theoretical, scientific, practical, or a combination.

---

## Step 2 — Identify the central thesis

Ask:

> Can the paper be summarized in one sentence?

If not, identify competing narratives.

---

## Step 3 — Examine the research gap

Ask:

> Does the paper establish a meaningful unresolved problem?

---

## Step 4 — Examine evidence

For each major claim:

```text
Claim
↓
Evidence
↓
Experiment
↓
Statistical support
↓
Alternative explanation
```

---

## Step 5 — Examine mechanism

If the paper explains why something happens:

> Does the evidence actually distinguish the proposed mechanism from plausible alternatives?

---

## Step 6 — Examine baselines

Ask:

> Are the baselines sufficient to support the conclusion?

---

## Step 7 — Examine robustness and failure

Ask:

> Does the paper establish where the approach works and fails?

---

## Step 8 — Examine scope

Ask:

> Are the conclusions broader than the evidence?

---

## Step 9 — Examine writing

Only after scientific issues are addressed, evaluate:

* structure,
* clarity,
* redundancy,
* terminology,
* transitions,
* concision,
* grammar.

---

# 43. Research Agent Revision Procedure

When revising a manuscript:

### First pass — Scientific structure

Check:

* research question,
* gap,
* hypothesis,
* contribution,
* evidence,
* interpretation.

### Second pass — Experimental logic

Check:

* baselines,
* metrics,
* ablations,
* robustness,
* statistics,
* failure analysis.

### Third pass — Narrative

Check:

* section order,
* argument progression,
* figure order,
* transitions.

### Fourth pass — Claims

Check:

* overstatement,
* unsupported causal language,
* generalization,
* novelty claims.

### Fifth pass — Writing

Check:

* clarity,
* concision,
* grammar,
* terminology,
* repetition.

Do not perform these passes in reverse order.

---

# 44. The 80% Removal Test

Ask:

> If 80% of the experiments disappeared, what single scientific conclusion should remain?

Then ask:

> Do the remaining experiments strongly support that conclusion?

If the answer is no, the paper may be trying to communicate too many ideas.

The solution may be:

* narrowing the research question,
* selecting a primary contribution,
* removing secondary experiments,
* or redesigning the evidence.

---

# 45. The Reviewer Memory Test

After reading the paper, imagine a reviewer is asked:

> What is this paper about?

A strong answer should identify a scientific question or insight.

Weak:

> They proposed a new model.

Stronger:

> They investigated why a particular learning objective succeeds at one type of structure but not another.

The exact answer depends on the research.

The key requirement is:

> **The paper should leave the reader with a clear scientific message.**

---

# 46. The "Why?" Ladder

When an important result appears, repeatedly ask:

> Why?

Example:

```text
Observation
↓
Why did this happen?
↓
Possible mechanism
↓
Why would that mechanism produce the observation?
↓
Underlying principle
↓
Can that principle be tested?
```

Stop when further "why" questions become unsupported speculation.

This process can reveal a stronger research question or motivate a targeted experiment.

---

# 47. The "What If This Is Wrong?" Test

For each major conclusion, ask:

> What alternative explanation could produce the same result?

Then determine whether the existing experiments distinguish between:

* the proposed explanation,
* confounding factors,
* measurement artifacts,
* implementation effects,
* dataset-specific effects,
* statistical noise,
* baseline differences.

If not, weaken the claim or propose a targeted experiment.

---

# 48. The "Simplest Explanation" Test

When interpreting a result, first consider simple explanations:

* data preprocessing,
* leakage,
* metric properties,
* model capacity,
* optimization,
* randomness,
* dataset bias,
* implementation details,
* baseline weakness.

Do not jump immediately to an elaborate theoretical explanation.

If a simple explanation is sufficient, acknowledge it.

---

# 49. Research Integrity

The agent must never:

* invent experimental results,
* invent citations,
* invent datasets,
* fabricate statistical significance,
* claim an experiment was performed when it was not,
* imply a result was preregistered when it was not,
* fabricate literature coverage,
* hide contradictory results,
* alter numbers to improve a narrative,
* or make unsupported claims of novelty.

If information is missing:

> State that it is missing.

Do not silently fill the gap.

---

# 50. Source Discipline

When working from supplied papers, datasets, notes, or other research materials:

* distinguish source-derived facts from inference,
* preserve source terminology when appropriate,
* do not silently change scientific meaning,
* identify contradictions,
* identify unsupported claims,
* and clearly separate evidence from interpretation.

When external literature is required, verify claims rather than relying on vague memory.

When evidence is unavailable, say so.

---

# 51. General Research Paper Checklist

Before finalizing a paper, verify:

## Scientific question

* [ ] Is the central question clear?
* [ ] Is it meaningful?
* [ ] Is it answerable?

## Research gap

* [ ] Is the gap precise?
* [ ] Is it supported by literature?
* [ ] Does the work address it?

## Contribution

* [ ] Is the primary contribution clear?
* [ ] Is it scientifically meaningful?
* [ ] Is it distinct from implementation details?

## Method

* [ ] Is the methodology reproducible?
* [ ] Are assumptions explicit?
* [ ] Is the method appropriately detailed?

## Experiments

* [ ] Are datasets appropriate?
* [ ] Are baselines relevant?
* [ ] Are metrics justified?
* [ ] Is the protocol clear?
* [ ] Is uncertainty reported?
* [ ] Are important confounders addressed?

## Results

* [ ] Does every major result answer a question?
* [ ] Is the main finding obvious?
* [ ] Are unexpected results addressed?
* [ ] Are failure modes investigated?
* [ ] Are ablations meaningful?

## Interpretation

* [ ] Are observations distinguished from explanations?
* [ ] Are causal claims supported?
* [ ] Are alternative explanations considered?

## Generalization

* [ ] Is the claim scope consistent with the evidence?
* [ ] Are limitations explicit?
* [ ] Are boundary conditions discussed?

## Writing

* [ ] Does each paragraph have a clear purpose?
* [ ] Are transitions logical?
* [ ] Is unnecessary material removed?
* [ ] Is the manuscript concise?
* [ ] Does the abstract match the actual findings?

## Final message

* [ ] Can the paper be summarized in one sentence?
* [ ] Can a reviewer identify the main scientific insight?
* [ ] Is that insight actually supported by the evidence?

---

# 53. Experiment Versioning & Version Run Archiving Guidelines for AI Agents

When assisting with iterative AI research, hypothesis testing, or benchmark expansions:

## 53.1 Never Overwrite Completed Research Runs
* **Preserve Completed Iterations**: Prior to launching a new experimental regime (such as Low-Data Label Efficiency, Out-of-Distribution scanner shift, or ablation studies), the agent must archive the previous run's checkpoints, metric CSVs, JSON logs, publication plots, and compiled manuscript PDF into a versioned subdirectory:
  ```text
  outputs/experiments/v{version}_{description}/
  ```
* **Explicit Tagging**: Use clear semantic versioning tags (e.g. `v1_full_data_100pct`, `v2_low_data_efficiency`, `v3_ood_generalization`).

## 53.2 Cross-Version Metric Comparisons
* Maintain a central manifest or script capable of comparing performance metrics across versions (`v1` vs `v2` vs `v3`).
* When reporting results to the user or writing paper revisions, explicitly cite which experiment version produced each table and figure.

---

# 54. Final Principle

The research agent should consistently prefer:

> **Question → Evidence → Explanation**

over:

> **Method → Benchmark → Numbers**

And:

> **Scientific understanding**

over:

> **Complexity for its own sake**

A strong research paper should not merely say:

> "Our method works."

It should explain:

> **What question was asked, why it matters, what was discovered, what evidence supports the discovery, why the result occurs, where the conclusion holds or fails, and what the research teaches us.**

The ultimate objective is not to make a paper sound impressive.

The objective is to make the scientific argument **clear, rigorous, reproducible, appropriately scoped, and useful to future researchers.**


