# Just me coding to learn world models

The implementation takes heavy inspiration from *Ha, D., & Schmidhuber, J. (2018). World models. arXiv preprint arXiv:1803.10122, 2(3).*

A few twitches make this implementation different from Dr.Ha's work:
- PyTorch instead of Tensorflow.
- Train/validate on MiniWorld environments.
- Use GRU instead of RNN for the memory model.
- Totally different vision model, memory model and controller model, because I'm just playing around.

## Some results
### Vision model
<table>
  <tr>
    <td><img src="./results/vision_model/img_0.png" width="300"/></td>
    <td><img src="./results/vision_model/img_1.png" width="300"/></td>    
    <td><img src="./results/vision_model/img_2.png" width="300"/></td>
  </tr>
  <tr>
    <td><img src="./results/vision_model/img_3.png" width="300"/></td>
    <td><img src="./results/vision_model/img_4.png" width="300"/></td>
    <td><img src="./results/vision_model/img_5.png" width="300"/></td>
  </tr>
  <tr>
    <td><img src="./results/vision_model/img_6.png" width="300"/></td>
    <td><img src="./results/vision_model/img_7.png" width="300"/></td>
    <td><img src="./results/vision_model/img_8.png" width="300"/></td>
  </tr>
</table>

