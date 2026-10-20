# Draft: email to the Surj authors

*To fill in before sending: the final numbers from `results.md` (section 4) and the
repository link once public. Authors: Yang & Howe, University of Washington. The address is in
the paper.*

---

**Subject:** Reproducing Surj (WWW'22) — code or a few implementation details?

Dear Dr. Yang and Prof. Howe,

We are reproducing *Ontological Learning for Fast, Accurate, and Robust Hierarchical
Multi-label Classification* (WWW'22 companion) as part of a benchmark study of hierarchical
multi-label classification on the C-HMCNN datasets. The paper says the code would be published
on GitHub, but we could not find it. Would you be willing to share it?

If the code is not available, these details would let us match your setup:

1. **Training:** batch size, number of epochs or the stopping rule, dropout, weight decay, and
   the optimizer. Was the final model trained on train only or on train + validation (as in
   C-HMCNN's script)?
2. **Network:** the sizes of the fully connected layers and the embedding dimension D. Did the
   classifier keep C-HMCNN's max-constraint module or MCLoss? The paper reports zero hierarchy
   violations on all 20 datasets. In our runs an unconstrained sigmoid head gives zero
   violations only with that module.
3. **Graph autoencoder:** the implementation (e.g. `tkipf/gae`), hidden/latent sizes, epochs,
   and whether all edges were used.
4. **Evaluation:** the exact AU(PRC) call, e.g. `average_precision_score(Y[:, to_eval], P[:,
   to_eval], average="micro")` from C-HMCNN, or another one. Were the root columns removed?
5. **Datasets without a validation file** (enron, diatoms, ImCLEF07): how was the validation
   set in Table 1 drawn?

For context, with C-HMCNN's own data pipeline (which reproduces C-HMCNN's published numbers to
three decimals in our hands), Kipf & Welling's GAE, and hyperparameters chosen on validation as
the paper describes, we obtain [FINAL NUMBERS] on FunCat (e.g. eisen [x] vs 0.392 reported,
cellcycle [x] vs 0.269). The gain over one-hot labels (Table 5) does reproduce in direction.
Every number, recipe and seed is in [REPOSITORY LINK].

Thank you for your time; any pointer would help.

Best regards,
[name, affiliation]
