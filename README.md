iornn-depparse
==============

A Lua implementation of the reranking-based dependency parser using inside-outside recursive neural networks described in

[1] Phong Le and Willem Zuidema (2014). The Inside-Outside Recursive Neural Network model for Dependency Parsing. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

Written and maintained by Phong Le (p.le [at] uva.nl)

###Package
This package contains three components

+ `source/` contains source code files in Lua of the IORNN reranker,

+ `tools/mstparser/` contains an extension of the [MSTParser 0.5.1](http://sourceforge.net/projects/mstparser/) which can now generate k-best candidates,

+ `data/wsj-dep/universal/dic/` contains the word list, POS list, dependent relation list and Collobert & Weston word embeddings for the experiments on the WSJ-U.


###Installation

Install [Torch7](torch.ch).

Compile the MSTParser following `tools/mstparser/README`.

###Usage

The following instruction is for replicating the results on the WSJ-U reported in the paper. Some small changes are needed for your own cases.


####Data

Convert the WSJ to dependencies using the [Universal Dependency Treebank Tool](http://code.google.com/p/uni-dep-tb/).

Split the treebank into train, test, dev portions, store them in `train.conll`, `test.conll`, `dev.conll` in the folder `data/wsj-dep/universal/data/`.


####Generate k-best lists
Execute

     cd tools/mstparser/
     ./train.sh #train the MSTParser on train.conll (you may need to change paths)
     ./kbest.sh #generate k-best lists for dev.conll and test.conll (you may need to change paths and K)


####Train the reranker
Default parameter values and file names are in `ds_spec.lua`.

Execute

    mkdir your_model_dir   
    nohup th train_depparse_rerank.lua ../data/wsj-dep/universal/dic ../data/wsj-dep/universal/data collobert your_model_dir 200 >& log_train &

A trained model after each epoch is stored in `your_model_dir`.

`200` is the net dimensions in our experiments.

All intermediate outputs are written down in `log_train`.

After each epoch, `dev.conll` is evaluated with UAS and LAS metrics. Pick the model that achieves the highest UAS (`model_29` in our experiments).


####Optimise K and alpha

Open `dp_spec.lua`, set `K_range = {1,10}`, `alpha_range = {0,1}`.

Execute

    th eval_depparse_rerank.lua your_model_path ../data/wsj-dep/universal/data/dev.conll ../data/wsj-dep/universal/data/dev-10-best-mst2ndorder.conll your_output

Pick `K` and `alpha` that achieve the highest UAS (9 and 0.68 in our experiments).


####Evaluation

Open `dp_spec.lua`, set `K` and `alpha` with the found values, set `K_range = nil`, `alpha_range = nil`.

Execute

    th eval_depparse_rerank.lua your_model_path ../data/wsj-dep/universal/data/test.conll ../data/wsj-dep/universal/data/test-10-best-mst2ndorder.conll your_output

You should get a UAS around 93.08%.



