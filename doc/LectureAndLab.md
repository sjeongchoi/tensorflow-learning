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


<hr />
# lab6
<hr />

Logistic regression의 경우 기본적으로 H(x) = WX
이것이 binary classification에 적합하지 않음
Z = H(x)로 놓고 G(z)가 큰 값들을 압축해서 0이나 1로 혹은 그 사이 값으로 나와주면 좋겠다고 가정
z축의 값이 커지더라도 1보다 커지지 않고, 아무리 작아져도 0보다 작아지지 않았으면 좋겠다로 나온 함수가 sigmoid(logistic)이 되었음
x->w->z->sigmoid->y^(y hat) 
H(x) -> y^ 

multinomial classification -> 클래스가 여러개 있다는 의미
두개의 값만 구분 가능한 binary classification을 가지고도 multinomial classification구현이 가능함
a, b, c의 경우가 있을 때
a와 그 이외의 값(b, c)
b와 그 이외의 값(a, c)
c와 그 이외의 값(a, b)
의 세개의 각각 다른 binary classification을 가지고 구분이 가능함


X -> A -> Y^
X -> B -> Y^
X -> C -> Y^

세개의 독립된 classification을 가지고 구분 가능~
각각의 행렬로 구할수 있음
  
각 세개의 행렬을 하나로 합칠 수 있음
-> matrix multiplication

각 값들이 각 가설의 값이 됨
Ha(X) Hb(X) Hc(X)
세개의 독립된 classification으로 하나의 벡터로 처리가 가능
바로 한 번에 계산이 가능

sigmoid를 하는 방법은 각각 적용해도 되지만 효율적으로 할 수도 있음

softmax function은 실수 값을 전체 합이 1이 되는 값으로 변화시켜 주는 것
tf.matmul(W, X)로 y의 값을 구한 후
softmax함수 값을 사용하여 값을 구해주면 됨

WX vs XW

x가 한개가 아니라 여러개가 되기 때문에
쉽게 처리하기 위해 하나의 식으로
W는 우리가 학습하는 데이터기 때문에 순서나 실제적인 행의 의미는 크게 상관 없어서 바꿈

cost function
cross entropy를 통해 구해서 최소화 시킴

gradientdescent후 minimize~


0 1.09774 [[ -8.33333252e-05   4.16666626e-05   4.16666480e-05]
 [  1.66666694e-04   2.91666773e-04  -4.58333408e-04]
 [  1.66666636e-04   4.16666706e-04  -5.83333429e-04]]
200 1.05962 [[-0.02051384 -0.00103983  0.02155367]
 [ 0.01406439  0.01097753 -0.0250419 ]
 [ 0.01431208  0.03574874 -0.05006079]]
400 1.04985 [[-0.04282598 -0.00625899  0.04908497]
 [ 0.01747187  0.00156368 -0.01903554]
 [ 0.01831204  0.04954104 -0.06785305]]
600 1.0407 [[-0.06517859 -0.01176361  0.07694223]
 [ 0.01943521 -0.00848972 -0.01094547]
 [ 0.0211558   0.06118308 -0.0823388 ]]
800 1.03194 [[-0.08734013 -0.01729389  0.10463406]
 [ 0.0211172  -0.01796171 -0.00315547]
 [ 0.02396628  0.07198346 -0.09594975]]
1000 1.02354 [[-0.10928574 -0.02282182  0.13210759]
 [ 0.02266255 -0.02678035  0.00411784]
 [ 0.02685851  0.08210357 -0.10896213]]
1200 1.01547 [[-0.13101467 -0.02834093  0.15935563]
 [ 0.02409704 -0.03497621  0.01087924]
 [ 0.02983199  0.091598   -0.12143002]]
1400 1.0077 [[-0.15252906 -0.03384716  0.1863762 ]
 [ 0.02543302 -0.04258838  0.01715543]
 [ 0.03287464  0.10050804 -0.13338274]]
1600 1.00021 [[-0.1738314  -0.03933693  0.21316831]
 [ 0.02668083 -0.04965464  0.02297391]
 [ 0.03597423  0.10887134 -0.14484566]]
1800 0.992979 [[-0.19492455 -0.04480689  0.23973146]
 [ 0.02784995 -0.05621057  0.02836075]
 [ 0.0391195   0.11672312 -0.15584266]]
2000 0.985988 [[-0.21581143 -0.05025397  0.26606542]
 [ 0.02894918 -0.06228961  0.03334056]
 [ 0.04230016  0.12409624 -0.16639641]]
----------------------------
[[ 0.46272627  0.35483009  0.18244369]] [0]
[[ 0.33820099  0.42101383  0.24078514]] [1]
[[ 0.27002314  0.29085544  0.4391214 ]] [2]
[[ 0.46272627  0.35483006  0.18244369]
 [ 0.33820099  0.42101383  0.24078514]
 [ 0.27002314  0.29085544  0.4391214 ]] [0 1 2]

Process finished with exit code 0


이번 실습에서 중요한 것은 data set을 training과 test로 나누는 것
반드시 데이터를 나누어야함
트레이닝 셋은 학습에만 사용 -> 모델에 학습시킴
테스트 셋의 경우, 모델의 입장에서는 한 번도 본 적이 없는 데이터로 학습시킨 모델을 평가하는데 사용

어떻게 나눌 것인가…

placeholder의 유용함이 여기에서 나옴
optimizer에는 트레이닝 데이터를 사용하고
prediction에는 테스트 데이터를 사용

Learning rate : NaN
Learning rate 이 너무 클 경우 밖으로 튕겨나감, 수렴이 되지 않고 발산이 됨
Learning rate 이 너무 작을 경우에는 학습이 굉장히 느리게 일어남
local minimun이 있을 경우, 거기에 빠질 수 있음


너무 클 경우(1.5)

(0, 5.7320299, array([[-0.30548954,  1.22985029, -0.66033536],
       [-4.39069986,  2.29670858,  2.99386835],
       [-3.34510708,  2.09743214, -0.80419564]], dtype=float32))
(1, 23.149357, array([[ 0.06951046,  0.29449689, -0.0999819 ],
       [-1.95319986, -1.63627958,  4.48935604],
       [-0.90760708, -1.65020132,  0.50593793]], dtype=float32))
(2, 27.279778, array([[ 0.44451016,  0.85699677, -1.03748143],
       [ 0.48429942,  0.98872018, -0.57314301],
       [ 1.52989244,  1.16229868, -4.74406147]], dtype=float32))
(3, 8.6680021, array([[ 0.12396193,  0.61504567, -0.47498202],
       [ 0.22003263, -0.2470119 ,  0.9268558 ],
       [ 0.96035379,  0.41933775, -3.43156195]], dtype=float32))
(4, 5.7711058, array([[-0.9524312 ,  1.13037777,  0.08607888],
       [-3.78651619,  2.26245379,  2.42393875],
       [-3.07170963,  3.14037919, -2.12054014]], dtype=float32))
(5, inf, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(6, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(7, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(8, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(9, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(10, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(11, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(12, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(13, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(14, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(15, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(16, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(17, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(18, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(19, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(20, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(21, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(22, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(23, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(24, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(25, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(26, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(27, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(28, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(29, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(30, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(31, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(32, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(33, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(34, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(35, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(36, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(37, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(38, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(39, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(40, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(41, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(42, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(43, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(44, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(45, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(46, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(47, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(48, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(49, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(50, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(51, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(52, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(53, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(54, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(55, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(56, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(57, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(58, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(59, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(60, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(61, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(62, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(63, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(64, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(65, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(66, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(67, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(68, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(69, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(70, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(71, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(72, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(73, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(74, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(75, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(76, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(77, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(78, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(79, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(80, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(81, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(82, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(83, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(84, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(85, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(86, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(87, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(88, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(89, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(90, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(91, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(92, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(93, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(94, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(95, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(96, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(97, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(98, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(99, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(100, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(101, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(102, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(103, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(104, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(105, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(106, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(107, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(108, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(109, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(110, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(111, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(112, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(113, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(114, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(115, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(116, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(117, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(118, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(119, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(120, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(121, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(122, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(123, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(124, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(125, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(126, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(127, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(128, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(129, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(130, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(131, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(132, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(133, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(134, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(135, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(136, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(137, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(138, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(139, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(140, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(141, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(142, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(143, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(144, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(145, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(146, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(147, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(148, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(149, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(150, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(151, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(152, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(153, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(154, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(155, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(156, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(157, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(158, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(159, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(160, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(161, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(162, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(163, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(164, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(165, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(166, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(167, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(168, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(169, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(170, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(171, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(172, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(173, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(174, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(175, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(176, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(177, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(178, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(179, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(180, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(181, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(182, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(183, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(184, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(185, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(186, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(187, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(188, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(189, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(190, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(191, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(192, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(193, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(194, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(195, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(196, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(197, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(198, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(199, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
(200, nan, array([[ nan,  nan,  nan],
       [ nan,  nan,  nan],
       [ nan,  nan,  nan]], dtype=float32))
('Prediction:', array([0, 0, 0]))
('Accuracy: ', 0.0)


1e-10

(0, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(1, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(2, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(3, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(4, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(5, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(6, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(7, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(8, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(9, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(10, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(11, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(12, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(13, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(14, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(15, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(16, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(17, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(18, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(19, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(20, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(21, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(22, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(23, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(24, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(25, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(26, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(27, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(28, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(29, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(30, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(31, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(32, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(33, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(34, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(35, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(36, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(37, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(38, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(39, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(40, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(41, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(42, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(43, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(44, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(45, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(46, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(47, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(48, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(49, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(50, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(51, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(52, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(53, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(54, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(55, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(56, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(57, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(58, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(59, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(60, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(61, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(62, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(63, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(64, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(65, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(66, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(67, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(68, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(69, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(70, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(71, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(72, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(73, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(74, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(75, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(76, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(77, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(78, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(79, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(80, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(81, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(82, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(83, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(84, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(85, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(86, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(87, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(88, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(89, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(90, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(91, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(92, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(93, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(94, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(95, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(96, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(97, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(98, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(99, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(100, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(101, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(102, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(103, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(104, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(105, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(106, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(107, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(108, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(109, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(110, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(111, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(112, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(113, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(114, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(115, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(116, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(117, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(118, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(119, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(120, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(121, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(122, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(123, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(124, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(125, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(126, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(127, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(128, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(129, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(130, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(131, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(132, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(133, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(134, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(135, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(136, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(137, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(138, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(139, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(140, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(141, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(142, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(143, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(144, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(145, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(146, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(147, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(148, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(149, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(150, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(151, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(152, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(153, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(154, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(155, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(156, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(157, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(158, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(159, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(160, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(161, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(162, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(163, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(164, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(165, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(166, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(167, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(168, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(169, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(170, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(171, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(172, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(173, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(174, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(175, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(176, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(177, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(178, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(179, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(180, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(181, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(182, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(183, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(184, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(185, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(186, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(187, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(188, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(189, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(190, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(191, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(192, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(193, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(194, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(195, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(196, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(197, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(198, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(199, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
(200, 5.7320299, array([[ 0.80269563,  0.67861295, -1.21728313],
       [-0.3051686 , -0.3032113 ,  1.50825703],
       [ 0.75722361, -0.7008909 , -2.10820389]], dtype=float32))
('Prediction:', array([0, 0, 0]))
('Accuracy: ', 0.0)

학습이 아예 일어나지 않는 경우에는 좀 크게 해줌

non normalized input
학습 rate를 잘 해주었는데도 nan발생할 경우 - 데이터가 노말라이즈 되지 않았을 수 있음
min-max scale 사용
0과 1사이의 값으로 모든 값을 normalize 해줌

의문 >>>>데이터가 들쭉날쭉한거는 어떻게 알 수 있을까?


mint

학습할 데이터가 크기 때문에 조금씩 잘라서 배치로 돌림

epoch : 전체 데이터 셋을 한 번 다 학습 시키는 것을 1epoch이라고 함
데이터가 많을 경우 다 메모리에 올리지 못하기 때문에 특정 개수만큼 자른 것을 batch size라고 함

총 데이터가 1000개 이고 배치 사이즈가 500 일 경우 2 에폭이 됨
