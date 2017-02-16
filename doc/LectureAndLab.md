# Machine Learning
<hr/>

python에 tensorflow를 import
.Session으로 세션을 시작

터미널에서 할 때 다음과 같은 에러가 발생

![Alt text](/Users/stella/Downloads/1.png)

이거는 그냥 cpu가 몇개인지 정의 안되었다는거라서 무시하고 하면 됨
위의 can't 어쩌고 하는 메시지는 CPU 코어 갯수를 정할 수 없어서 4개로 가정한다는 의미이므로 예제를 실행하는데는 무리가 없다.
실제 기계학습(Machine Learning) 을 진행할 때에는 장비의 CPU 스펙을 고려하여 조정하면 되겠다.


출처: http://migom.tistory.com/16 [devlog.gitlab.io]

텐서 플로우는 모든 것이 오퍼레이션 
상수 자체가 오퍼레이터가 됨
노드를 시작하는 것은 run으로 시작


>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 4
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 4
>>> print(sess.run(hello))
Hello, TensorFlow!




>>> a = tf.constant(2)
>>> b = tf.constant(3)
>>> c = a + b
>>> print c
Tensor("add:0", shape=TensorShape([]), dtype=int32)
>>> print sess.run(c)
5

단순히 더해서 프린트 한거에서는 노드의 형태만 가지고 있는것 실제 실행은 안되어있음

세션 run을 실행시켜야 원하는 결과를 볼 수 있음



>>> with tf.Session() as sess:
...      print "a = 2 b = 3"
...      print "Addition with constants : %i" % sess.run(a+b)
...      print "Multiplication with constants: %i" % sess.run(a*b)
... 
a = 2 b = 3
Addition with constants : 5
Multiplication with constants: 6



플레이스 홀더 : 함수 같은 것을 사용할 때 파라미터를 줌..
텐서 플로우에서는 타입만 정의해주면 함수에 전해주는 파라미터를 만들 수 있음
모델이 함수는 아니지만 실행 시점에서 값들을 넣을 수 있게 해주는 기능.


>>> a = tf.placeholder(tf.int16)
>>> b = tf.placeholder(tf.int16)
>>> add = tf.add(a, b)
>>> mul = tf.mul(a, b)
>>> with tf.Session() as sess:
...     print "Addition with variables : %i" % sess.run(add, feed_dict={a: 2, b: 3})
...     print "Multiplication with variables : %i" % sess.run(mul, feed_dict={a: 2, b:4})
... 
Addition with variables : 5
Multiplication with variables : 8


linear regression

여러 데이터를 가지고 학습을 시킴 : training
학습을 통하여 모델을 생성 -> 학습이 완료
regression은 학습된 모델을 기반으로 값을 예측하여 알려줌..
이를 linear regresion이라 함


레그레이션 모델을 학습다는 것은 가설을 세운다는 전제..
리이너한 모델이 우리 데이터에 맞을 것이다
세상에 있는 많은 현상들이 linear한 경우로 많이 나옴~
데이터가 있다면 그에맞는 linear한 선을 찾는다는 것이 학습을 한다는 것
H(x) = Wx + b => 리니어의 가장 기본적인 가설

H(x) -> 가설
선의 모양은 W와 b에 따라 다름

어떤 선이 우리가 가지고 있는 데이터에 가장 맞는 선인가 정해야함

더 좋은 가설을 알 수 있는 방법 : 실제 데이터와 가설이 나타내는 예측 값과의 차이를 계산 => cost(loss) function
가설과 실제 데이터의 차이를 계산 (H(x) - y)^2 차이가 클 때 더욱 패널티를 줄 수 있음~
cost = 
