# Final development-only regularizer selection (2026-09-06)

The initial seed-zero search and boundary expansion revealed unstable fits at
large primitive-MSE weights. To avoid presenting one favorable development
initialization as adequate tuning, fix a final common grid before new fits:
weights {0.1,1,10,100,1000,10000}, all three penalties, development seeds 0,1,2.
Select on the mean of the same five path accuracies across all three dev seeds;
tie break toward smaller weight. Fit seeds 3,4 only for the selected weights.
Do not omit collapsed runs. Evaluate the chosen models at both test lengths,
and preserve every earlier selection/result as history. This is the FINAL
grid for this campaign, not an invitation to optimize against test outcomes.

All original outcomes and follow-up test data have already been inspected.
Selection itself uses development only, but the campaign is post-hoc and cannot
claim an untouched confirmation dataset. No optimizer budget or recorded data
changes. Report total distinct fits/compute across the complete search history.
