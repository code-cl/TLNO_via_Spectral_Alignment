# Cross-geometry transfer learning for neural operators via spectral alignment and adaptive fine-tuning
We present a spectral alignment framework that enables efficient cross-geometry transfer learning for neural operators. First, we construct a shared spectral representation by computing aligned spectral bases for source and target geometries via approximate joint diagonalisation. These aligned spectral bases are integrated with a frequency-domain neural operator to train a source model for source geometry. To adapt this model to the target geometry with very limited data, we introduce a dedicated two-stage fine-tuning strategy: spectral alignment is first refined, followed by fine-tuning model parameters.
![image](https://github.com/code-cl/TLNO_via_Spectral_Alignment/blob/main/TLNO_Framework.png)


