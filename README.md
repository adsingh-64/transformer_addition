# transformer_addition
Transformer does 3-digit addition!

The transformer is a vanilla decoder-only transformer -- the main difficulty in 
building this model was presenting the data. We want the transformer to learn 
the right to left addition algorithm that one is taught in grade school. Hence,
for the task of 3-addition, we use a maximum context size of 6 tokens, and feed 
in the addition of two numbers in a consistent manner with the right to left
algorithm.

For example, the problem 723 + 599 is fed in to the transformer as 
[3, 9, 2, 9, 7, 5]. The label for [3, 9] is [2], the label for [2, 9] is [2] (carryover),
and the label for [7, 5] is [13] (carryover). That is, the transformer predicts the digits in reverse,
so that its output is flipped during inference when testing whether or not it gets addition problems correct.
The logits corresponding to the labels for [3], [3, 9, 2], and [3, 9, 2, 9, 7] are nonsensical in terms
of the right to left addition algorithm and hence are ignored by the cross entropy loss.

The major difficulty with very small transformers was the carryover rule.
Theoretically, the necessary and sufficient context is there for the 
transformer to see whether it needs to carryover or not. However, we observed a 
grokking effect where only once the transformer was bumped up to its size
as in transformer_addition.py was it able to develop a proper hidden representation
of the addition problem that allowed it to master the carryover rule. The
transformer in transformer_addition.py achieves 99.9% accuracy on 3-digit addition
(tested over 10000 problems) after 10 minutes of training on 1x A100 GPU.

This observation suggests that 3-digit (or n-digit addition) is a very interesting
case for mechanistic interpretation (see Quirke and Barez, 2024, https://arxiv.org/pdf/2310.13121). 


