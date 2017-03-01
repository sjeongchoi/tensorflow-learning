# Machine Learning
<hr/>

python에 tensorflow를 import
.Session으로 세션을 시작

터미널에서 할 때 다음과 같은 에러가 발생

![Alt text](https://github.com/sjeongchoi/tensorflow-learning/blob/master/1.png?raw=true)

이거는 그냥 cpu 코어 개수를 정할 수 없어 4개로 가정한다고 한다는 말이라서 무시하고 하면 된다고 함.

위의 can't 어쩌고 하는 메시지는 CPU 코어 갯수를 정할 수 없어서 4개로 가정한다는 의미이므로 예제를 실행하는데는 무리가 없다.
실제 기계학습(Machine Learning) 을 진행할 때에는 장비의 CPU 스펙을 고려하여 조정하면 되겠다.
출처: http://migom.tistory.com/16 [devlog.gitlab.io]


텐서 플로우는 모든 것이 오퍼레이션 
상수 자체가 오퍼레이터가 됨
노드를 시작하는 것은 run으로 시작

<pre><code>
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 4
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 4
>>> print(sess.run(hello))
Hello, TensorFlow!
</code></pre>

<pre><code>
>>> a = tf.constant(2)
>>> b = tf.constant(3)
>>> c = a + b
>>> print c
Tensor("add:0", shape=TensorShape([]), dtype=int32)
>>> print sess.run(c)
5
</code></pre>

단순히 더해서 프린트 한거에서는 노드의 형태만 가지고 있는것 실제 실행은 안되어있음

세션 run을 실행시켜야 원하는 결과를 볼 수 있음

<pre><code>
>>> with tf.Session() as sess:
...      print "a = 2 b = 3"
...      print "Addition with constants : %i" % sess.run(a+b)
...      print "Multiplication with constants: %i" % sess.run(a*b)
... 
a = 2 b = 3
Addition with constants : 5
Multiplication with constants: 6
</code></pre>

플레이스 홀더 : 텐서 플로우에서는 타입만 정의해주면 함수에 전해주는 파라미터를 만들 수 있음
모델이 함수는 아니지만 실행 시점에서 값들을 넣을 수 있게 해주는 기능.

<pre><code>
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
</code></pre>

<hr/>
# linear regression
<hr/>

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

<hr />
# LAB2
<hr />

먼저 트레이닝 데이터부터 정의
W와 b의 값을 텐서 플로우에서 지정하는 variable로 지정 -> 나중에 이 모델이 업데이트를 할 수 있기 때문에 variable로 지정해야함
tensorflow가 잘 찾을 수 있는지 랜덤한 값을 생성

cost는 h - y값을 뺀 값을 square를 한 다음 평균을 구함
minimize는 blackbox..cost를 작게하여 minimize함

변수를 초기화 후 실행시켜 주어야함 > 초기화 안하면 에러 메시지 나옴
minimize는 for 문으로 2000번 돌면서…
계속 실행 시켜줌

<pre><code>
>>> import tensorflow as tf
>>> x_data = [1, 2, 3]
>>> y_data = [1, 2, 3]
>>> W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
>>> b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
>>> hypothesis = W * x_data + b
>>> cost = tf.reduce_mean(tf.square(hypothesis - y_data))
>>> a = tf.Variable(0.1)
>>> optimizer = tf.train.GradientDescentOptimizer(a)
>>> train = optimizer.minimize(cost)
>>> init = tf.initialize_all_variables()
>>> sess = tf.Session()
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 4
can't determine number of CPU cores: assuming 4
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 4
>>> sess.run(init)
>>> for step in xrange(2001):
...   sess.run(train)
...   if step % 20 == 0:
...     print step, sess.run(cost), sess.run(W), sess.run(b)
... 
</code></pre>

0 0.394947 [ 0.58223546] [ 1.36335051]

20 0.091077 [ 0.64948994] [ 0.79679173]

40 0.0344114 [ 0.78454977] [ 0.48976901]

60 0.0130016 [ 0.86756784] [ 0.30104935]

80 0.00491234 [ 0.9185971] [ 0.18504788]

...

660 4.73695e-15 [ 0.99999994] [  1.48965142e-07]

680 0.0 [ 1.] [  5.35976952e-08]

700 0.0 [ 1.] [  5.35976952e-08]

...

2000 0.0 [ 1.] [  5.35976952e-08]


각 스탭마다의 cost와 W, b값을 출력
cost가 0 가까이 가게 됨


place holder : 처음에 값을 넣어주는게 아니라 place holder 를 선언한 다음 그것을 모델에 사용하고, operation에 사용한 다음 실제로 실행할 때 > run이라는 명령을 할 떄  변수들을 입력
placeholder의 이점 : *****가지고 있는 모델의 재활용 가능*****

<pre><code>
>>> x_data = [1., 2., 3.]
>>> y_data = [1., 2., 3.]
>>> W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
>>> b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
>>> X = tf.placeholder(tf.float32)
>>> Y = tf.placeholder(tf.float32)
>>> hypothesis = W * X + b
>>> cost = tf.reduce_mean(tf.square(hypothesis - Y))
>>> a = tf.Variable(0.1)
>>> optimizer = tf.train.GradientDescentOptimizer(a)
>>> train = optimizer.minimize(cost)
>>> init = tf.initialize_all_variables()
>>> sess = tf.Session()
>>> sess.run(init)
>>> for step in xrange(2001):
...   sess.run(train, feed_dict={X:x_data, Y:y_data})               ...   if step % 20 == 0:
...     print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)
... 

>>> print sess.run(hypothesis, feed_dict={X:5})
[ 5.00000143]
>>> print sess.run(hypothesis, feed_dict={X:2.5})
[ 2.50000024]
</code></pre>

<hr />
# lab3 minimizing cost
<hr />

import tensorflow as tf
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32);
Y = tf.placeholder(tf.float32);

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(20):
  sess.run(update, feed_dict={X:x_data, Y:y_data})
  print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
</code></pre>

0 0.219411 [ 1.21683347]

1 0.0624105 [ 1.11564457]

...

18 3.20976e-11 [ 1.00000262]

19 9.54969e-12 [ 1.00000143]

<hr />
# lab 4 multi-variable linear regression
<hr />

가설에 변수가 2개가 들어오게 됨
>> 각각에 대해 학습할 weight가 2개가 됨
tensorflow에서도 w가 각각 학습되어야함

변수가 늘어남에 따라 같은 방식으로 확장할 수 있음

<pre><code>
import tensorflow as tf
x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
  sess.run(train)
  if step % 20 == 0:
    print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)

</code></pre>


0 0.480649 [ 0.93938279] [ 0.61236995] [ 1.00284827]

20 0.0491891 [ 0.86178893] [ 0.83601344] [ 0.5257743]

...

500 2.55795e-14 [ 0.99999988] [ 0.99999994] [  2.41646831e-07]

520 2.27374e-14 [ 1.] [ 0.99999994] [  1.98731541e-07]

...

1980 1.42109e-14 [ 1.] [ 0.99999994] [  1.74889706e-07]

2000 1.42109e-14 [ 1.] [ 0.99999994] [  1.74889706e-07]

# matrix

복잡한 수식을 행렬을 통해 간단하게 바꿀 수 있음
행렬로 바꾸는거 역시 간단한
x, W가 차원이 생김 > 2차원 배열로 됨
행렬 곱셈을 해야함 > 각각의 앨리먼트 곱해서 더해나가야함 (tensorflow의 matmul)

0 1.74255 [[ 0.72554004  1.15851843]] [ 1.16975248]

20 0.0557844 [[ 0.82536542  0.85281485]] [ 0.55991417]

40 0.0161519 [[ 0.90603071  0.92080081]] [ 0.30128455]

...

1980 4.83169e-14 [[ 0.99999994  0.99999994]] [  1.63099003e-07]

2000 4.83169e-14 [[ 0.99999994  0.99999994]] [  1.63099003e-07]


b값을 없애서 더 간단하게 수식으로 표현 가능
전체 데이터에 1을 추가함

<pre><code>

import tensorflow as tf
x_data = [[1, 1, 1, 1, 1],
          [0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
  sess.run(train)
  if step % 20 == 0:
    print step, sess.run(cost), sess.run(W)
</code></pre>

0 0.162583 [[ 0.3971453   0.8274079   1.03055084]]

20 0.0074302 [[ 0.20434552  0.93626559  0.94628346]]

...

200 1.06256e-07 [[  7.72808155e-04   9.99758959e-01   9.99796808e-01]]

220 3.07795e-08 [[  4.15828748e-04   9.99870300e-01   9.99890625e-01]]

...

480 2.27374e-14 [[  2.22217537e-07   9.99999940e-01   1.00000000e+00]]

500 1.42109e-14 [[  1.74533866e-07   9.99999940e-01   1.00000000e+00]]

...

2000 1.42109e-14 [[  1.74533866e-07   9.99999940e-01   1.00000000e+00]]


데이터가 많아지고 종류가 많아지면 소스에 하드코딩으로 박아 넣기 어려움 >  파일 형태로 읽는 것은 어떨까?
> numpy를 사용하여 txt파일을 loadtxt 할 수 있음

<hr />
# linear 
<hr />
regression  : linear로 가설을 세움

hypothesis : H(x)  = WX

Cost : 가지고 있는 학습 데이터와 가설의 선이 얼마나 차이나는지 평균을 구하는 것

학습을 하다는 것은 우리가 가지고 있는 데이터를 이용해서 cost를 최소화 하는 weight를 찾아내는 것

밥그릇을 엎어놓은 모양으로 gradient decent사용 가능 :  어떤 점에서 시작하던지 최저점을 찾을 수 있음 : 우리가 찾고자 하는 weight의 값

기울기는 cost를 미분한 값

얼마나 움직일지는 a로 주어진 running weight값

<hr />
# logistic
<hr />

classification -> binary classification

둘 중 하나로 정해진 카테고리 중에 고르는 것

spam detection(spam or ham)

credit card fraudulent transaction detection(legitimate/fraud)


많이 활용되는 분야임 : 0/1 encoding

linear와 비슷해 보이지만 x가 아무리 커져도 결론은 1 혹은 0

linear는 기울기를 가진 선형으로 값이 1보다 크거나 0보다 작은 값이 나와야하는데 결과는 그 값이 나오지 않음


선형이 좋기는 하나 0과 1사이로 값이 나올 수 있는 형태로 바꾸어주어야함

G(Z) = 1/(1+e^-z) : sigmoid

z가 커질 수록 G(Z)는 1에 수렴

z가 작아질수록 G(Z)는 0에 수렴

z = WX

H(X) = G(Z)

logistic hypothesis

H(X) = 1 / (1 + e^(-WX))



cost function

가설이 sigmoid로 바뀌면서 값이 0에서 1사이에 들어오도록 변경

-> 그려보면 울퉁불퉁한 형태로 바뀜 : 시작점에 따라 최저점이 다르게 나올 수 있음(local minimum) 찾고자 하는 것은 global minimum

cost학습이 local minimun에서 멈추면 제대로 된 값을 얻을 수 없음

-> cost(w) = 1/m(sum(c(H(x), y)))

c(H(x), y) = -log(H(x)) : y = 1
             -log(1-H(x)) : y = 0

e을 사용하니까 대척점에 있는 log를 사용

예측이 맞으면 0, 틀리면 무한에 수렴

g(x) = -log(x)
 
y = 1 (-log(x))

H(x) = 1 -> cost(1) = 0

H(x) = 0 -> cost(0) = 무한대

* 예측을 틀리면 cost가 굉장히 높아지게 됨

y = 0 (-log(1-H(x)))

H(x) = 0 ->  cost(0) = 0

H(x) = 1 ->  cost(1) = 무한대

원래 목적이었던 cost에 맞는 함수를 찾을 수 있게 됨

두개의 함수가 매끈해지니까 저 두개를 붙인 후 경사 하강 알고리즘을 사용할 수 있음

c(H(x),y) = -ylog(H(x)) - (1 - y)log(1 - H(x))

minimize cost - Gradient decent algorithm

cost(W) = -1/m(sum(c(H(x),y) = -ylog(H(x)) - (1 - y)log(1 - H(x))))

W := W - a(s/sW)*cost(W)

<hr />
# lab5
<hr />
logistic regression

hypothesis : H(X) = 1 / (1 + e^(-WX))

cost : cost(W) = -1/m(sum(c(H(x),y) = -ylog(H(x)) - (1 - y)log(1 - H(x))))

weight : W := W - a(s/sW)*cost(W)



0 0.879362 [[-0.80477178  0.87826937 -0.6495688 ]]

20 0.608889 [[-1.00166404  0.50610757 -0.0928071 ]]

40 0.511841 [[-1.20049405  0.24702492  0.19963428]]

60 0.469687 [[-1.38790584  0.1136347   0.37520102]]

...

1940 0.18168 [[-7.69624996  0.22889401  1.7213006 ]]

1960 0.180996 [[-7.73245955  0.22935809  1.72887194]]

1980 0.180319 [[-7.76846933  0.22981085  1.73640788]]

2000 0.17965 [[-7.80428457  0.23025247  1.74390912]]





0 0.681543 [[ 0.31453696 -0.517407    0.64153147]]

20 0.631313 [[ 0.02564165 -0.40226352  0.57970673]]

40 0.59059 [[-0.24397732 -0.31425875  0.54691768]]

60 0.556688 [[-0.49573141 -0.24824996  0.53549677]]

80 0.527761 [[-0.73115361 -0.19843049  0.5385924 ]]

100 0.502639 [[-0.95177227 -0.15999182  0.5508498 ]]

...

1960 0.183858 [[-7.58235645  0.2273747   1.69752753]]

1980 0.183149 [[-7.61921072  0.22787634  1.70521295]]

2000 0.182448 [[-7.65585375  0.22836553  1.71286118]]

[[False]]
[[ True]]
[[False  True]]

