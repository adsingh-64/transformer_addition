# transformer_addition
Transformer does 3-digit addition!
The transformer is uninteresting -- the main difficulty in building this model
was presenting the data. We want the transformer to learn the right to left
addition algorithm, so we feed the data in as ...

Carryover was a major problem, theoretically context is there for transformer
to pick up on this, but was difficult with smaller models and had to bump
up the transformer to larger size before it got good at carryover,
grokking effect.

