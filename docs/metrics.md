# ActiveMap Metrics

ActiveMap reports executable map outcomes, not only action classification. This
file defines the metrics used by the current paper package and open-source
evaluation scripts.

## Primary Endpoint

**Final map quality** measures the committed map after the controller and Safe
Commit decision. It is computed after executing the typed vector writeback and
rerasterizing or graph-evaluating the result. For SN7 and SpaceNet-style
polygon tasks, this is usually raster IoU or polygon IoU. For MUNO21 roads, it
is reported with APLS and Pixel-F1.

Metric direction: higher is better.

## Safety Metrics

**False edit rate** is the rate at which the system writes a change when the
safe outcome should preserve the prior or avoid the proposed edit.

Metric direction: lower is better.

**Missed edit rate** is the rate at which the system fails to write a real
required edit. Safe Commit can reduce false edits by increasing missed edits,
so both must always be reported together.

Metric direction: lower is better.

**Wrong edit rate** captures committed edits with the wrong typed operation or
invalid operation semantics.

Metric direction: lower is better.

## Cost Metrics

**Additional evidence rate** is the fraction of episodes where the controller
uses optional evidence beyond the direct candidate.

Metric direction depends on matched quality/safety; lower is better only when
quality and safety are not worse.

**Additional cost** is the sum of registered downstream evidence/tool costs.
It does not include the shared candidate-perception front end in the current
paper protocol.

## Utility Metrics

Utility profiles combine map quality, false edits, missed edits, and evidence
cost for model selection or diagnostic comparisons. Utility is never reported
alone as proof of improvement; component metrics must be shown beside it.

Generic form:

```text
utility =
  w_quality * final_map_quality
  - w_false * false_edit_rate
  - w_missed * missed_edit_rate
  - w_cost * additional_cost
  - w_invalid * invalid_geometry_rate
```

The exact profile used by a run must be recorded in the config or aggregate
receipt.

## Factorial Contrasts

For the V5 matched non-KEEP audit, the key validation-only contrasts are:

```text
selection factor =
  selected_safe_commit - direct_safe_commit

Safe Commit factor =
  selected_safe_commit - selected_commit

forced cost control =
  selected_safe_commit - forced_safe_commit
```

Promotion requires the predeclared quality, safety, and cost gates. A positive
point estimate is not enough when the lower confidence bound is not strictly
positive.

## Bootstrap And Units

- SN7: bootstrap by AOI, with controller/updater seed preserved as an explicit
  source of variation when applicable.
- MUNO21: bootstrap by task scenario or held-out road scene.
- SpaceNet 8: bootstrap by held-out active episode when support is sufficient.
- Multi-seed summaries must state whether intervals are seed-then-unit,
  unit-only after seed aggregation, or diagnostic ranges.

## Reporting Rules

- Always report absolute metrics for the main policies, not only deltas.
- Report false edits and missed edits together.
- Mark validation-only results as validation-only.
- Mark extension/boundary results that fail promotion gates.
- Do not average incompatible metrics across polygon, road, and disaster
  tasks into a single headline score.
