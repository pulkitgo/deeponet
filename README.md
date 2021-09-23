# DeepONets

This implementation is based on the code in the [DeepONet repository.](https://github.com/luluxvi/deeponet)

For running on Google Colab, append the following cell  before running the code:

```
%tensorflow_version 2.x
!pip install pathos
!export DDEBACKEND=tensorflow.compat.v1
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```
