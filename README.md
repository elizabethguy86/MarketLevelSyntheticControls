## Store-Level Synthetic Control

Fits a separate synthetic control for **each individual test unit** and returns the original dataframe with an `expected` column (or `expected_{outcome}` columns for multiple outcomes) representing each test store's counterfactual trajectory.

Unlike `SyntheticDiD`, which treats all test units as one pooled treatment group, `StoreLevelSyntheticControl` fits independent unit weights per unit, so each unit gets its own tailored synthetic control. This allows for breakouts of various groups of stores that ladder up to the overall effect.

Other test units are **excluded from each units's donor pool** to prevent contamination between treated units.
