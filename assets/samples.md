# Samples from the `configs/tiny.yaml` run

Model: 869,504 params (4 layers, 4 heads, n_embd 128, block 128, vocab 512), trained for
3000 steps on an Apple M3 CPU (146.9 s). Final val loss 2.860 nats/token.

Each sample is the verbatim output of the command shown (temperature 0.8, top-k 50,
200 new tokens, KV-cache decoding). Seeds 1, 2 and 3 were fixed in advance and each
command was run once; the model is small, so expect Shakespeare-shaped nonsense, not Shakespeare.

## Prompt: `ROMEO:`

```sh
uv run tinygpt sample --ckpt out/tiny/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --top-k 50 --seed 1
```

```text
ROMEO:
I thernext my liege, still I warrant it,
Nor hither, so, both distrest I am the comfort.
If thou dost fill'd, thou take my soul adverse
Or, seeing me a man incount; which, which, if
An honour and reath by Verona's face, youthful daughter
To marriorge you full thinks with doubt,
It mine eyes to our marriage's defence
With purposed the tedious briends.

GLOUCESTER:

KING EDWARD IV:
So stay
```

## Prompt: `JULIET:`

```sh
uv run tinygpt sample --ckpt out/tiny/ckpt.pt --prompt "JULIET:" --max-new-tokens 200 --temperature 0.8 --top-k 50 --seed 2
```

```text
JULIET:
O, imaged thou wast minister'd the after.

CLIFFORD:
If thou be a clim the wind of England,
Madam, Himprint, here he light and claim bars
That great coldly for in heaven swords upon,
Or, the sight--flolder, whose thanks't, whose reply.

WARWICK:
Why, I love a chaste, come to quarrel,
And gentlemen, a mother, that heavy dog,
As that my unbears to get thee serve, and m
```

## Prompt: `First Citizen:`

```sh
uv run tinygpt sample --ckpt out/tiny/ckpt.pt --prompt "First Citizen:" --max-new-tokens 200 --temperature 0.8 --top-k 50 --seed 3
```

```text
First Citizen:
We may aloney no more will be born to orne
cours a pretty manyes made.

BENVOLIO:
I would not be found thy country. I must be
In Romeo moved. Away, indignation!

ROMEO:
You may not be sleep, sir, for your wife's news
To whom it doth make you see the land.

MERCUTIO:
Yea, I am to see, then?

COMINIUS:
Welcome.

MENENIUS:
This is no more.

MENENIUS:
You're we
```
