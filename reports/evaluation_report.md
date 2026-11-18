# RAG Evaluation Report

## 1. Evaluation Setup

The RAG system was evaluated using a dataset of 100 questions
created by five team members, with 20 questions per author.

The dataset includes six question categories:
content, administrative, cross-course, image-based,
follow-up, and unanswerable questions.

The questions were divided into:
- Development set: 40 questions
- Test set: 60 questions

All 100 questions passed validation with zero errors
and zero warnings.

Development experiments were used to compare configurations,
identify errors, and guide system improvements.

After freezing the system at v1.2-eval, B0, B1, and Final
were each evaluated once on the 60 held-out test questions.

Development and test results are reported separately.

Three configurations were compared:

- B0: No RAG. The language model answers without retrieving course documents.
- B1: Dense retrieval with prompt v1 and syllabus-only sources.
- Final v1.1: The updated RAG configuration with the complete
  retrieval and generation pipeline.

All three configurations were evaluated on the same 40 development questions.

## 2. Quantitative Results

| Metric | B0 (No RAG) | B1 | Final v1.1 |
|---|---:|---:|---:|
| Recall@K | N/A | 34.29% | 94.29% |
| MRR@K | N/A | 0.2676 | 0.7167 |
| Refusal accuracy (automatic) | 87.5% | 20.0% | 85.0% |
| False refusal rate | 0.0% | 91.43% | 8.57% |
| Correct unanswerable refusal rate | 0.0% | 100.0% | 40.0% |
| Citation presence | 0.0% | 7.5% | 87.5% |
| Citation validity | N/A | 100.0% | 100.0% |
| Exact correctness (automatic) | 15.0% | 12.5% | 55.0% |
| Mean correctness score (automatic) | 16.25% | 13.75% | 57.5% |
| Mean latency | 1.81 s | 1.40 s | 1.97 s |

All three evaluation runs completed all 40 questions without
execution failures.

Recall and MRR are not applicable to B0 because no retrieval
is performed. B0's automatic refusal accuracy of 87.5%
reflects correct non-refusal decisions on 35 answerable
questions, while all five unanswerable questions were answered.

### Manual Review of Final v1.1

The automatic judge marked 22 of 40 answers fully correct (55.0%).

Manual review identified five additional correct answers:

- FRA17, IRE17, SER17: valid semantic refusals incorrectly
  classified as non-refusals.
- FRA05: correct answer penalized because the reference answer
  omitted Management Control and Time Series.
- FRA09: correct explanation of the Bertrand paradox.

After these documented corrections, reviewed exact correctness
is 27/40 (67.5%).

The automatic and manually reviewed results are reported
separately. SA13 remains pending manual review.

## 3. Main Findings

The final system substantially outperformed B1 in retrieval
quality, answer correctness, and citation presence.

Recall@K increased from 34.29% to 94.29%, while exact
correctness increased from 12.5% to 55.0%.

The false refusal rate decreased from 91.43% to 8.57%.

However, the final system required more processing time,
increasing mean latency from 1.40 to 1.97 seconds.

These results demonstrate a trade-off between response quality
and computational cost.

## 4. Error Analysis

The automatic evaluation identified 18 incorrect answers
in the final configuration.

The initial classification was:

- 12 potential generation errors
- 3 false refusals
- 3 incorrect non-refusals

Manual inspection revealed that the three incorrect
non-refusals were actually valid semantic refusals.

The model explicitly acknowledged that the requested
information was unavailable, but the automatic evaluator
did not recognize the alternative refusal wording.

After documenting these three manual corrections:

- Semantic refusal accuracy: 92.5%
- Correct unanswerable refusal rate: 100%
- False refusal rate: 8.57%

Two additional judge errors were confirmed during manual
review: FRA05 and FRA09. Together with the three semantic
refusal corrections, these increase reviewed exact correctness
to 67.5% (27/40).

The original automatic metrics were preserved separately
to ensure reproducibility.

## 5. Qualitative Error Analysis

### Image-based questions

FRA13:
The correct Hitters regression-tree slide was retrieved,
including the root split "Years < 4.5".
However, the generated answer provided a general explanation
of recursive binary splitting instead of identifying
the specific threshold.

SA13:
The correct Arrogant Brewery brand-image VRIO slide was
retrieved. The answer included the essential correct facts
but added unrelated information from other VRIO diagrams.

SA13 received a judge score of zero and remains flagged
for manual review.

### Follow-up questions

EL15:
The evaluation runner correctly passed the conversation
history. However, the generated answer repeated the
Bertrand equilibrium explanation without comparing it
with the Cournot example requested in the follow-up.

### Cross-course questions

IRE05 and SA10 were incorrectly refused.

Subsequent development debugging identified two causes:
a mismatch between American and British spelling of
Organizational Behaviour, and cross-course questions that
were restricted to the course table without supplementary
retrieval. The team has implemented fixes, but these changes
are not reflected in the v1.1 results reported here.

Other remaining errors, including EL02, SA06, SA15, and EL15,
highlight model limitations in factual accuracy and
conversational reasoning.

## 6. Evaluation Limitations

Several limitations must be considered.

First, the evaluation uses 40 development questions.
Results may differ on the held-out test set.

Second, LLM-based correctness judgments may contain errors
and should be complemented with manual review.

Third, retrieval metrics measure whether retrieved chunks
cover annotated gold source IDs. This does not guarantee
that the model uses the evidence correctly.

Fourth, the automatic refusal detector relies on an exact
refusal phrase, potentially misclassifying valid semantic
refusals.

Finally, the reported experiments compare B0, B1, and
Final v1.1 on development data only. These comparisons do
not constitute a complete ablation study. The held-out
test set remains untouched pending the frozen v1.2 configuration.

## 7. Final Held-Out Test Evaluation

### 7.1 Evaluation Protocol

After development evaluation and configuration tuning, the
team froze the system at v1.2-eval.

The three configurations (B0, B1, and Final v1.2) were each
executed once on the same 60 held-out test questions.

Each execution completed all 60 questions without technical
failures.

Factual correctness was evaluated using the qwen2.5:7b judge
with scores of 0, 0.5, or 1.

The results below are automatic evaluation results without
manual corrections to the test set.

### 7.2 Test Results

| Metric | B0 (No RAG) | B1 (Basic RAG) | Final v1.2 |
|---|---:|---:|---:|
| Questions evaluated | 60 | 60 | 60 |
| Exact correctness | 11.67% | 26.67% | 55.00% |
| Mean correctness score | 15.83% | 26.67% | 58.33% |
| Recall@K | N/A | 28.89% | 88.89% |
| MRR@K | N/A | 0.2000 | 0.7444 |
| Refusal accuracy | 75.00% | 38.33% | 80.00% |
| False refusal rate | 0.00% | 80.00% | 6.67% |
| Correct unanswerable refusal rate | 0.00% | 93.33% | 40.00% |
| Citation presence | 0.00% | 16.67% | 85.00% |
| Citation validity | N/A | 100.00% | 100.00% |
| Mean latency | 3.04 s | 1.56 s | 2.16 s |

Recall@K and MRR@K are not applicable to B0 because it
does not retrieve documents. Citation validity is also
not applicable because B0 generates no citations.

B0 never refused a question. Its 75% refusal accuracy
reflects 45 correct non-refusal decisions on answerable
questions, while all 15 unanswerable questions were answered.

### 7.3 Correctness Comparison

The exact numbers of fully correct answers were:

- B0: 7/60 (11.67%).
- B1: 16/60 (26.67%).
- Final v1.2: 33/60 (55.00%).

Final v1.2 improved exact correctness by 43.33 percentage
points compared with B0 and 28.33 percentage points
compared with B1.

Its Recall@K also increased from 28.89% in B1 to 88.89%,
indicating substantially better retrieval performance.

However, retrieval success did not always translate into
factually correct answers.

### 7.4 Development vs. Test Performance

| Metric | Final v1.2 Dev (40) | Final v1.2 Test (60) |
|---|---:|---:|
| Exact correctness | 67.50% | 55.00% |
| Mean correctness | 68.75% | 58.33% |
| Recall@K | 97.14% | 88.89% |
| MRR@K | 0.7452 | 0.7444 |
| Refusal accuracy | 95.00% | 80.00% |
| Citation presence | 92.50% | 85.00% |
| Mean latency | 1.94 s | 2.16 s |

Final v1.2 performed less accurately on the held-out test
set than on development questions.

This difference highlights the importance of evaluating
the frozen system on previously unseen questions.

### 7.5 Remaining Limitations

The final system still has limitations.

First, its 88.89% test Recall@K is higher than its 55%
exact correctness, showing that retrieving relevant
evidence does not guarantee a correct generated answer.

Second, the automatic correct unanswerable refusal rate
is 40%. Handling questions without available evidence
remains an important area for improvement.

Third, citation validity measures whether citations point
to valid retrieved sources. It does not establish whether
each cited source supports the associated factual claim.

Finally, response latency depends on the hardware and
execution environment. The measurements reported here
were obtained from the local evaluation runs.

## 8. Conclusion

The evaluation demonstrates that the final RAG system
substantially outperforms both baselines on held-out questions.

Final v1.2 achieved 55.00% exact correctness on test,
compared with 26.67% for B1 and 11.67% for B0.

Retrieval performance also improved substantially:
Recall@K increased from 28.89% in B1 to 88.89% in Final.

These improvements demonstrate the value of the enhanced
retrieval and generation pipeline.

Nevertheless, the system still exhibits limitations in
factual answer generation and refusal decisions.

In particular, high retrieval recall does not guarantee
correct answers, and the automatic correct unanswerable
refusal rate remains at 40%.

The evaluation therefore supports the effectiveness of
the final system while identifying clear opportunities
for future improvement.
