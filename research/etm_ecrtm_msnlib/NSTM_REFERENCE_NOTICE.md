# NeuralSinkhornTopicModel reference notice

`benchmarks/neural_ms2lda/nstm.py` is a PyTorch adaptation of the model
equations and Sinkhorn procedure published in He Zhao et al., *Neural Topic
Model via Optimal Transport* (ICLR 2021), using the authors' reference source:

- <https://github.com/ethanhezhao/NeuralSinkhornTopicModel>
- pinned commit: `610d1604d5467289028714ed0ce684dfb5ef8a7b`

The original repository is distributed under the following MIT license.

> MIT License
>
> Copyright (c) 2021 He Zhao
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

TopMost commit `ef24433859b2e283959ddef7f95020a40abb104f` was used only as an
independent numerical cross-check; its source is not vendored here.
