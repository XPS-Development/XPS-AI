# Post-processing: workflow and implementation

The formulas below use LaTeX math notation, rendered by GitHub and Markdown viewers
with math support.

This document describes the **current implementation** of the Post-processing mode in
XPS-AI. It covers the user workflow, matching algorithm, analytical calculations,
persistence, and application boundaries. It is an implementation reference rather
than a roadmap.

## 1. Purpose and scope

Post-processing works with spectra that already have peak decompositions. It connects
components across an explicitly ordered spectrum series, exposes the resulting peak
trends, and builds analytical tables from selected trends belonging to one or more
series.

Three objects have different responsibilities:

| Object | Responsibility |
| --- | --- |
| Spectrum series | An ordered list of spectrum IDs, matching settings, manual constraints, and the last matching snapshot. |
| Peak trend | A continuous chain of matched components within one series; called a `PeakTrack` internally. |
| Analytical table | An independent document object containing selected trend sources, aligned rows, calculation definitions, and plot selections. |

A series can feed several tables. A table can combine several series. Deleting a table
does not delete its source series or decompositions.

Matching and calculations do not refit peaks, modify measured arrays, infer chemical
states, assign doublet roles, or apply sensitivity factors. Trend names are user labels.
Area ratios are model-area ratios, not atomic concentrations.

## 2. User workflow

### Create and inspect a series

1. Import spectra and create their decompositions in **Decomposition** mode.
2. Switch to **Post-processing** and click **New** under **Spectrum series**.
3. Filter the file/group/spectrum tree and select spectra. The picker displays only
   spectra containing peaks; a checked parent selects only descendants visible under
   the current filter.
4. Set the name and inspect the ordered steps. Reorder them by dragging or using
   **Up** and **Down**. Tree names are sorted lexically, so measurement order must be
   checked explicitly when names contain numeric prefixes.
5. Save the series. Matching runs synchronously and automatically; creation and its
   initial result form one undoable change.

The sidebar contains separate **Spectrum series** and **Analytical tables** lists.
Click either list to open its object. Double-click an object's name to edit it inline;
renaming supports undo/redo.

The series workspace has three tabs:

| Tab | Behavior |
| --- | --- |
| **Trend components** | Select a trend and inspect its components at every series step. The compact view shows step, spectrum, position, Area, sig, Rel. area, and link status. **Details** exposes additional metadata. Internal IDs stay hidden in the main view. |
| **Trend preview** | Plot the selected trend's numerical values. Position, Area, sig, and relative area have separate panels. Missing values produce gaps. |
| **Neighbouring links** | Inspect all trends on a common energy axis and edit links between two adjacent steps. The trend selector and trend-specific actions are hidden on this tab. |

Double-clicking a component row or clicking a plotted source point opens its original
decomposition through the controller. The trend component table supports view sorting
without changing the saved spectrum order or source identity.

**Name trend…** gives a track a meaningful label. Unnamed tracks have short labels such
as `Trend 1`; their internal IDs remain available for references and export.

### Review neighbouring links

The default view is a waterfall of the **Whole series**. **Neighbouring pair** limits
the plot to the two spectra selected in the pair dropdown.

Measured curves share one binding-energy axis, decreasing from left to right. Each
spectrum is shifted vertically. White fill below a nearer curve masks curves behind
it, producing an overlapping stack. **Vertical step** controls the spacing relative
to the displayed spectrum height; the default is `0.18`.

**Normalize heights** scales each spectrum independently for display. Otherwise the
stack uses a shared intensity scale. These transformations do not change stored data.
**Decomposition** and **Peak links** enable optional overlays. Peak markers select the
editable components; **Assign link** requires the chosen connection and **Break link**
forbids it. Each action recalculates matching in the same undo step.

When the whole series is already rendered, changing the pair or selected peaks updates
highlighting without reevaluating curves, recreating the plot, or resetting zoom.
Overlay switches change item visibility. Switching the displayed pair in pair-only
mode, changing stack geometry, or changing matching/decomposition inputs requires a
redraw.

### Build an analytical table

Use **Add trend to table…** or create a table from the sidebar. **Sources…** selects
specific trends, each qualified by its owning series, and shows their coverage and
component composition.

Rows pair source series **by supplied step index**. For example, row 3 combines step 3
from every selected series. There is no cross-series matching by energy, timestamp,
or spectrum name. The longest source series determines the row count. An absent peak
or shorter series leaves an explicit missing point; later rows never shift to fill it.
Unequal source lengths or peak counts produce a warning rather than silently dropping
rows.

Add calculation columns with explicit source operands. Columns can be named, edited,
removed, reordered, selected for plotting, and exported. Analytical tables deliberately
have **no row sorting**: header clicks and programmatic sorting do not change row order.
Graph points and cells retain references to the contributing spectra and components.

## 3. Values, calculations, and units

The matching adapter reads physical, unnormalized component DTOs from every region of
each selected spectrum. Backgrounds and malformed components are recorded as exclusions.
Zero-area peaks are retained; there is no area threshold for eligibility.

| Value | Definition and presentation |
| --- | --- |
| Position | The model's `cen` parameter, displayed in `eV`. |
| sig | The model's positive `sig` parameter, displayed in `eV`. It is not converted to FWHM. |
| Area | The area returned by the model's area evaluation, preserved numerically and displayed without a unit label. No RSF correction is applied. |
| Rel. area | The peak area as a percentage of all eligible peak areas in that spectrum. |

For peak $i$ in spectrum step $t$, let $A_{i,t}$ be its model area and
$\mathcal{P}_t$ the set of all eligible peaks in that spectrum. Relative area is:

$$
R_{i,t} = 100\,\frac{A_{i,t}}{\displaystyle\sum_{j\in\mathcal{P}_t} A_{j,t}}
\qquad [\%].
$$

FWHM is not calculated, used for matching, offered as a calculation, or shown in
post-processing tables and plots. An optional `fwhm` descriptor field remains only
for reading saved prototype snapshots. New adapter output leaves it unset.

Plots group columns by **quantity and unit**, not by unit alone. Consequently sig and
position have different panels even though both use `eV`. Automatic SI prefixes are
disabled on the value axes: sig stays in eV instead of becoming meV, and Area is not
shown with an artificial `area` or `karea` unit. A custom column name does not change
its underlying plotted quantity.

The supported calculation grammar is fixed:

| Operation ID | Calculation |
| --- | --- |
| `position`, `sig`, `area` | Read one selected source's value. |
| `spectrum_area_fraction` | Relative area against all eligible peaks in the source spectrum. |
| `position_difference` | Position A minus position B in the aligned row. |
| `position_shift` | Current position minus the same source's position at an explicit reference row. |
| `area_sum` | Sum the areas of an explicit source set. |
| `area_ratio` | Area A divided by area B. |
| `area_sum_ratio` | Sum of numerator areas divided by sum of denominator areas. |
| `area_fraction` | Percentage of one source's area within an explicit denominator set containing that source. |

For selected sources $a$ and $b$, reference row $t_0$, and explicit source sets
$\mathcal{N}$ and $\mathcal{D}$, the corresponding calculations are:

$$
\begin{aligned}
\Delta p_{a,b,t} &= p_{a,t} - p_{b,t}, \\
\delta p_{a,t;t_0} &= p_{a,t} - p_{a,t_0}, \\
S_{\mathcal{N},t} &= \sum_{i\in\mathcal{N}} A_{i,t}, \\
Q_{a,b,t} &= \frac{A_{a,t}}{A_{b,t}}, \\
Q_{\mathcal{N},\mathcal{D},t}
  &= \frac{\displaystyle\sum_{i\in\mathcal{N}} A_{i,t}}
           {\displaystyle\sum_{j\in\mathcal{D}} A_{j,t}}, \\
R_{a,\mathcal{D},t}
  &= 100\,\frac{A_{a,t}}{\displaystyle\sum_{j\in\mathcal{D}} A_{j,t}}
  \qquad [\%],\quad a\in\mathcal{D}.
\end{aligned}
$$

Here $p$ denotes position. The full-spectrum relative area uses all eligible peaks
of one spectrum; the selected-source fraction uses only the explicit denominator set.

Repeated operands within a sum are counted once. Area sums have no displayed unit;
ratios are dimensionless and fractions are percentages. Arbitrary formula text and
chains of calculated-column operands are not supported.

Cells carry a value, status (`ok`, `ambiguous`, or `missing`), reason, and source
provenance. Missing operands, missing peaks, unavailable spectra, absent features,
nonfinite results, and zero denominators produce missing cells with explicit reasons.
A sum with a missing required operand remains missing rather than becoming a partial
sum. Calculations retain full precision; rounding is an export option.

## 4. Matching algorithm

### Input and constraints

`MatchingSeries` contains immutable ordered steps and physical descriptors. A descriptor
stores component and region IDs, model name, position, area, sig, and optional `frac`
under the internal name `shape_gl`. Region IDs provide provenance, not matching gates.
Empty and unavailable steps remain in the input.

An edge can connect a component only to a component in the **next supplied step**.
Every component has at most one predecessor and one successor. Every eligible peak
belongs to a track, including a singleton. Tracks cannot bridge an empty intermediate
step, split, or merge.

Manual constraints are directed component pairs:

- Required pairs must be selected and bypass the automatic energy gate.
- Forbidden pairs cannot be selected.
- Both endpoints must exist in adjacent steps. A pair cannot be both required and
  forbidden, and required pairs cannot introduce competing predecessors or successors.

### Transition cost

For a candidate edge $a \rightarrow b$, let $p$ denote position, $\sigma$ the
model's `sig`, $g$ its `frac`, $f$ the nonnegative area fraction, and $r$ the
normalized position rank. The transition cost is:

$$
\begin{aligned}
C(a,b) ={}& w_E\,H\!\left(\frac{p_b-p_a}{s_E}\right) \\
          &+ w_\sigma\,H\!\left(\frac{\ln(\sigma_b/\sigma_a)}{s_\sigma}\right) \\
          &+ w_g\,H\!\left(\frac{g_b-g_a}{s_g}\right) \\
          &+ w_A\left(\sqrt{f_b}-\sqrt{f_a}\right)^2
           + w_r\left(r_b-r_a\right)^2.
\end{aligned}
$$

The robust penalty is the Huber loss with a unit transition point:

$$
H(x) =
\begin{cases}
\dfrac{x^2}{2}, & |x| \leq 1, \\
|x|-\dfrac{1}{2}, & |x| > 1.
\end{cases}
$$

The weights $w_E,w_\sigma,w_g,w_A,w_r$ correspond to `energy_weight`,
`width_weight`, `shape_weight`, `area_weight`, and `rank_weight`. The scales
$s_E,s_\sigma,s_g$ correspond to `energy_scale`, `width_scale`, and `shape_scale`.

For matching, areas are clipped to nonnegative values before normalization:

$$
f_{i,t} =
\begin{cases}
\dfrac{\max(0,A_{i,t})}{\displaystyle\sum_{j\in\mathcal{P}_t}\max(0,A_{j,t})},
  & \displaystyle\sum_{j\in\mathcal{P}_t}\max(0,A_{j,t}) > 0, \\
0, & \text{otherwise}.
\end{cases}
$$

Position rank is normalized within a step, with tied positions receiving their
average rank. These matching features are distinct from the displayed relative-area
calculation: clipping is part of the matching feature, not a modification of model areas.

Automatic edges outside `energy_gate` are rejected. Optional sig/shape terms are
omitted when a feature is unavailable and the missing feature is recorded when its
weight is enabled. Missing features are not replaced with guessed values.

Current defaults from `MatchingSettings` are:

| Setting | Default |
| --- | ---: |
| `energy_scale` | 0.75 eV |
| `energy_gate` | 4.0 eV |
| `energy_weight` | 1.0 |
| `width_scale` | 0.3, for the logarithmic sig ratio |
| `width_weight` | 0.0 |
| `shape_scale` | 0.2 |
| `shape_weight` | 0.0 |
| `area_weight` | 0.1 |
| `rank_weight` | 0.1 |
| `start_cost` | 10.0 |
| `end_cost` | 10.0 |
| `ambiguity_threshold` | 0.5 |

Thus sig and shape are available matching features but are disabled by default.

### Path cover and ambiguity

Let $\mathcal{L}$ be the selected links and $\mathcal{T}$ the resulting tracks.
The matcher minimizes:

$$
J(\mathcal{L}) = \sum_{(a,b)\in\mathcal{L}} C(a,b)
  + |\mathcal{T}|\left(\lambda_{\mathrm{start}}+\lambda_{\mathrm{end}}\right),
\qquad |\mathcal{T}| = N-|\mathcal{L}|.
$$

Here $N$ is the number of eligible peaks, while $\lambda_{\mathrm{start}}$ and
$\lambda_{\mathrm{end}}$ are `start_cost` and `end_cost`.

It reduces the minimum-cost path cover to a rectangular assignment problem and solves
it with SciPy's `linear_sum_assignment`. For `N` peaks, the matrix has `N` rows and
`2N` columns, including dummy unmatched choices. Candidate edge costs are reduced by
the saved start/end costs. Required edges fix their assignment row and target column.
Components are processed in sorted ID order for deterministic tie resolution.

For each selected automatic edge, the solver runs again with that edge forbidden.
Let $J^*$ be the original minimum and $J^*_{\neg e}$ the minimum with automatic
edge $e$ forbidden. Its cost margin and ambiguity test are:

$$
m_e = \max\!\left(0, J^*_{\neg e}-J^*\right),
\qquad e\text{ is ambiguous}\iff m_e\leq\tau,
$$

where $\tau$ is `ambiguity_threshold`. Required manual links
have no estimated margin. A track is ambiguous if any of its links is ambiguous;
that status propagates to affected projected values.

The implementation uses a dense assignment matrix and additional solves for ambiguity.
Runtime depends on the number of components and candidate links, not just the number
of spectra. Execution is synchronous; there is no background worker or cancellation UI.

## 5. State, identity, and invalidation

`PostProcessingSeries` persists its definition, settings, constraints, names, last
`MatchingResult`, input fingerprint, and revision token. The SHA-256 input fingerprint
covers spectrum order/availability, region bounds, model identity, physical parameter
values/bounds/expressions/vary flags, settings, and constraints. Display names are not
matching inputs.

A result is stale when no snapshot exists or its fingerprint differs from current
inputs. Editing decompositions or series configuration retains the old result for
inspection and undo; **Match peaks** recalculates it. Creating a series and editing a
manual link already run matching automatically. Renaming alone does not stale matching.

Tracks use an anchor component and an internal `track/<anchor>` identity. Names are
stored against anchors. Recalculation preserves existing identities where possible
and protects anchors used by names or analytical-table sources. A disappeared protected
anchor or a merged chain containing several protected anchors raises an explicit
conflict instead of silently transferring a name or operand.

Analytical tables persist:

- Qualified `TrendSource` records containing series, track, and anchor IDs.
- Stable `AlignedRow` IDs with explicit `PointReference` records for each source.
- `TableMetric` definitions, column order, plotted columns, and dependency stamps.

Realignment reuses row IDs where shared source-spectrum references still agree.
Reference calculations use these row IDs rather than visible row numbers. Dependency
stamps include matching inputs and saved results but exclude display names. Recalculate
changed sources and use **Update alignment** to accept their new correspondence.
Deleting a source leaves unresolved references available for diagnosis and undo.
Stale or conflicting tables cannot be exported as current results.

All persistent edits use application changes and commands, including rename, source
selection, formulas, plot selection in analytical tables, and manual links. Undo/redo
restores complete saved states. Series/track selections, zoom, waterfall spacing, and
visible waterfall overlays are view state rather than document calculation inputs.

## 6. Architecture and source map

```text
Qt widgets -> ControllerWrapper -> AppOrchestrator -> use cases -> Change -> commands
                    |                                      |
                    +---- QueryService <- immutable DTOs <- core projections
```

The orchestrator executes and forwards; widgets do not evaluate models or mutate the
core stores. Core code has no Qt or application-layer dependency.

| Source | Responsibility |
| --- | --- |
| [ui/post_processing/panel.py](../ui/post_processing/panel.py) | Series/table navigation, workspace tabs, commands, and source activation. |
| [ui/post_processing/series_dialog.py](../ui/post_processing/series_dialog.py) | Filtered peak-bearing spectrum selection and explicit step order. |
| [ui/post_processing/source_dialog.py](../ui/post_processing/source_dialog.py), [metric_dialog.py](../ui/post_processing/metric_dialog.py) | Qualified trend sources and explicit calculation operands. |
| [ui/post_processing/links.py](../ui/post_processing/links.py) | Waterfall rendering, manual-link controls, selection updates, and plot caching. |
| [ui/post_processing/table.py](../ui/post_processing/table.py), [trends.py](../ui/post_processing/trends.py) | Table presentation, stable source navigation, quantity-specific plots, and units. |
| [app/usecases/post_processing.py](../app/usecases/post_processing.py) | Series creation, synchronous matching, guarded result application, and anchor preservation. |
| [app/usecases/analytical_tables.py](../app/usecases/analytical_tables.py) | Independent tables, source changes, alignment, and calculation definitions. |
| [app/post_processing.py](../app/post_processing.py), [app/query_service.py](../app/query_service.py) | Physical input snapshots, operation metadata, projections, and stale/conflict queries. |
| [core/post_processing/adapter.py](../core/post_processing/adapter.py), [types.py](../core/post_processing/types.py) | Eligibility checks and immutable matching inputs/results. |
| [core/post_processing/matching.py](../core/post_processing/matching.py) | Continuous costs, constraints, assignment, path extraction, and ambiguity margins. |
| [core/post_processing/state.py](../core/post_processing/state.py), [fingerprint.py](../core/post_processing/fingerprint.py) | Persistent series state and decomposition invalidation. |
| [core/post_processing/tables.py](../core/post_processing/tables.py), [metrics.py](../core/post_processing/metrics.py) | Per-series projections, quantities, operation validation, and numerical metrics. |
| [core/post_processing/analysis.py](../core/post_processing/analysis.py) | Table state, dependency stamps, row alignment, cross-series evaluation, and cell provenance. |
| [core/dto.py](../core/dto.py) | Shared immutable rows, columns, cells, and source locations. |
| [app/command/changes.py](../app/command/changes.py), [commands.py](../app/command/commands.py) | `SetPostProcessingSeries` / `SetAnalyticalTable` changes and undoable state replacement. |
| [core/io/post_processing.py](../core/io/post_processing.py), [serialization.py](../core/io/serialization.py) | Saved sections, validation, and document loading. |
| [core/io/post_processing_csv.py](../core/io/post_processing_csv.py), [app/usecases/export.py](../app/usecases/export.py) | Shared-projection CSV export, provenance sidecars, and export guards. |

`AnalysisColumn.quantity` separates quantities with the same unit. Analytical views
use `QIdentityProxyModel` with sorting disabled; component views can use
`QSortFilterProxyModel` and resolve navigation back to source rows. The link editor
keys its render cache by document generation, series revision, current input
fingerprint, and spectrum names. Selection changes reuse existing graphical items.

## 7. Saving and export

The current document format is **1.2**. Its separate `post_processing` and
`analytical_tables` sections preserve definitions, result snapshots, explicit
references, and dependency information. Plain JSON and gzip documents are supported.
Loading is transactional: invalid input must not partially overwrite the live document.
Append operations remap colliding identities while retaining internal references.

The reader also accepts release-format **1.0** documents without post-processing
sections. Intermediate format **1.1** is rejected; there is no migration of its old
table/trend representation.

CSV export consumes the same full-precision projections as the UI and graphs. It
includes source IDs, column IDs/units, values, and per-cell status/reason fields.
A sibling `<name>.calculations.json` records definitions, precision, column metadata,
source-series snapshots, and the model-area basis. Optional decimal precision changes
written values only, not stored calculations. Hidden UI IDs remain available in export
for traceability.

## 8. Verification and current limits

The main regression suites are:

- [tests/core/post_processing](../tests/core/post_processing): adapter eligibility,
  continuous matching, constraints, ambiguity, model sig, quantities, calculations,
  missing-value behavior, and row alignment.
- [tests/app/usecases/test_post_processing.py](../tests/app/usecases/test_post_processing.py)
  and [test_analytical_tables.py](../tests/app/usecases/test_analytical_tables.py): automatic
  creation, undo/redo, protected identities, invalidation, and reusable table sources.
- [tests/core/io/test_post_processing.py](../tests/core/io/test_post_processing.py):
  round trips, unresolved/stale state, compatibility, append collisions, and transactional load.
- [tests/ui/post_processing](../tests/ui/post_processing): filtered selection, navigation,
  inline naming, fixed tab positioning, waterfall/hover interaction, unchanged zoom on
  selection, explicit units, separate sig/position plots, and disabled analytical sorting.

Run the focused suites with:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/core/post_processing \
  tests/app/usecases/test_post_processing.py \
  tests/app/usecases/test_analytical_tables.py \
  tests/core/io/test_post_processing.py \
  tests/ui/post_processing
```

Current limits are adjacent-step matching, positional cross-series alignment, the
fixed calculation grammar, and synchronous dense assignment. There is no automatic
chemical interpretation, time-axis inference, gap bridging, concentration correction,
or uncertainty propagation. Waterfall normalization is a display aid; it is not a
normalization of model areas used in calculations.
