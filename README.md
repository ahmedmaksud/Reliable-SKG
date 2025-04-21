# Secret Key Generation by Continuous Encryption Before Quantization

For detiled description please refer to our [SPL22](https://intra.ece.ucr.edu/~yhua/Reprint_Maksud_Hua_SPL_2022.pdf) paper.

## Introduction
Secret Key Generation (SKG) is a long standing problem for network security applications. For wireless security, a pair of nodes (Alice and Bob) in a wireless network can exploit their reciprocal channel state information to generate a secret key. Such a key shared by Alice and Bob can be then used as a symmetric key for information encryption between them over any networks.
A central issue of SKG is how to best transform a pair of highly correlated secret vectors (SVs) at Alice and Bob respectively into a pair of nearly identical sequences of binary bits (i.e., keys). The SVs are in practice quasi-continuous (due to finite precision of real number representation). Since the two SVs collected at Alice and Bob are generally not equal, the probability of the generated keys being unequal, i.e., key error rate (KER), is generally nonzero. So a central objective of SKG is to minimize KER.
The major steps of SKG for both wireless security and biometric security are: extraction of SVs which should be maximally correlated with each other and contain the minimal amount of non-secret, quantization of SVs with KER as small as possible, and reconciliation and privacy amplification for improved key. In this project, we focus on the problem of quantization to turn a pair of SVs into a pair of keys with any length, small KER and sufficient randomness.
