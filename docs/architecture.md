# Architecture (Summary)

PRESTO models the antigen processing → pMHC binding → presentation → TCR recognition
pipeline using shared transformer encoders and biologically constrained heads.

Highlights:

- Processing is MHC-independent.
- Binding is modeled via three kinetic latents (stability, intrinsic, chaperone).
- Presentation combines processing + binding.
- TCR recognition combines pMHC and TCR embeddings.
- Immunogenicity requires both presentation and recognition.

For full details, see `ARCHITECTURE.md` in the repository.
