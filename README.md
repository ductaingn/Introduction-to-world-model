# Just me coding to learn world models

The implementation takes heavy inspiration from *Ha, D., & Schmidhuber, J. (2018). World models. arXiv preprint arXiv:1803.10122, 2(3).*

A few twitches make this implementation different from PhD.Ha's work:
- PyTorch instead of Tensorflow.
- Train/validate on MiniWorld environments.
- Use GRU instead of RNN for the memory model.
- Totally different vision model, memory model and controller model, because I'm just playing around.

## Some results
### Vision model
![](./results/vision_model/img_0.png)
![](./results/vision_model/img_1.png)
![](./results/vision_model/img_3.png)
![](./results/vision_model/img_4.png)
![](./results/vision_model/img_5.png)
![](./results/vision_model/img_6.png)
![](./results/vision_model/img_7.png)
![](./results/vision_model/img_8.png)
![](./results/vision_model/img_9.png)
![](./results/vision_model/img_10.png)
