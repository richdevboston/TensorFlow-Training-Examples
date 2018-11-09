# -*- coding: utf-8 -*-
""" Created on Thu Nov  8 23:24:58 2018  @author: lysakowski """

''' tensorboard --logdir="C:\ProgramData\examples_CODE\output2" --port 6006 '''

#%%
'''Test that TensorFlow is running correctly'''
import tensorflow as tf 

Zeebie = tf.constant([10,100], name='const_Zeebie')

#%% 
import tensorflow as tf
p = tf.constant(0.034)
c = tf.constant(1000.0)
x = tf.add(c, tf.multiply(p, c))
x = tf.add(x, tf.multiply(p, x))
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output2", sess.graph)
    print(sess.run(x))
    writer.close()

#%% 

'''Fetches and Feed Dictionary
We give fetches and feed_dict pass into every "session.run" command. 
The Fetches parameter indicates what we want to compute. 
The feed dictionary specifies the placeholder values for that computation.'''

import tensorflow as tf
W = tf.constant([10,100], name='const_W')

#these placeholders can hold tensors of any shape
#we will feed these placeholders later

x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

#tf.multiply is simple multiplication and not matrix
Wx = tf.multiply(W,x, name="Wx")
y = tf.add(Wx,b, name='y_Add')

with tf.Session() as sess2:

	'''All the code which require a session is writer here.
	Here Wx is the fetches parameter. Fetches refers to the node of the graph 
	we want to compute feed_dict is used to pass the values for the placeholders
	'''
	print( "Intermediate result Wx: ", sess2.run(Wx, feed_dict={x: [3,33]}))
	print( "Final results y: ",        sess2.run(y, feed_dict={x:[5,50],b:[7,9]}))
	
	writer = tf.summary.FileWriter('output2',sess2.graph)
	
	writer.close()


#%% ----------------------------------------
'''Variables: Variables are mutable tensor values that persist across 
multiple calls to sesssion.run(). '''

import tensorflow as tf
W2 = tf.Variable([2.5, 4.0], tf.float32, name='var_W2')

# W is a Variable
x2 = tf.placeholder(tf.float32, name='x2')
b2 = tf.Variable([5.0,10.0], tf.float32, name='var_b2')

# b is also a variable with initial value 5 and 10
y2 = W2 * x2 + b2

#initialize all variables defined
init = tf.global_variables_initializer()

# Global_variable_initializer() will declare all the variable we have initialized
# use with statement to instantiate and assign a session

with tf.Session() as sess:
	sess.run(init)
	#this computation is required to initialize the variable
	print("Final result: Wx + b = ", sess.run(y2,feed_dict={x2:[10,100]}))

	# changing values 
	number = tf.Variable(2)
	multiplier = tf.Variable(1)
	init = tf.global_variables_initializer()
	result = number.assign(tf.multiply(number,multiplier))

with tf.Session() as sess:
	sess.run(init)

	for i in range(10):
		print("Result number * multiplier = ", sess.run(result))
		print("Increment multiplier, new value = ", sess.run(multiplier.assign_add(1)))


	writer = tf.summary.FileWriter('output2',sess2.graph)
	
#	writer.close()
	
	
#%% ----------------------------------------
'''Multiple graphs for one TensorFlow program

We can explicitly create as many graphs inside a TensorFlow program. Any TensorFlow program have a default graph which contains all the placeholders and variables you have instantiated. But we can logically segment the graph by instantiating a graph explicitly using tf.graph() . Below program may answer some of your doubts.
'''

import tensorflow as tf
g1 = tf.Graph()

'''set g1 as default to add tensors to this graph using default methord'''
with g1.as_default():
	with tf.Session() as sess:
		A = tf.constant([5,7],tf.int32, name='A')
		x = tf.placeholder(tf.int32, name='x')
		b = tf.constant([3,4],tf.int32, name='b')
		y = A * x + b

		print( sess.run(y, feed_dict={x: [10,100]}))
		writer = tf.summary.FileWriter('output2',sess.graph)

'''to ensure all the tensors and computations are within the graph g1, we use assert'''
assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
	with tf.Session() as sess:
		Q = tf.constant([5,7],tf.int32, name='Q')
		v = tf.placeholder(tf.int32, name='v')
		yhat = tf.pow(Q, v, name='yhat')
		print( sess.run(yhat, feed_dict={v: [3,5]}))
		writer = tf.summary.FileWriter('output2',sess.graph)

assert yhat.graph is g2

'''same way you can access default graph '''
default_graph = tf.get_default_graph()

with tf.Session() as sess:
	C = tf.constant([5,7],tf.int32, name='C')
	zeebie = tf.placeholder(tf.int32, name='zeebie')
	yfoot = C + zeebie
	print(sess.run(yfoot, feed_dict={zeebie: [3,5]}))
	writer = tf.summary.FileWriter('output2',sess.graph)

assert yfoot.graph is default_graph

#%% ----------------------------------------
'''Named Scope:  TensorBoard may be most extremely useful debugging tool, but as your graph explodes in size, you need ways to see in a bigger picture. 
Now run the below program using TensorFlow and view its graph in TensorBoard.'''

import tensorflow as tf

A = tf.constant([4], tf.int32, name='A')
B = tf.constant([4], tf.int32, name='B')
C = tf.constant([4], tf.int32, name='C')
x = tf.placeholder(tf.int32, name='x')

# y = Ax^2 + Bx + C
Ax2_1 = tf.multiply(A, tf.pow(x,2), name="Ax2_1")
Bx = tf.multiply(A,x, name="Bx")
y1 = tf.add_n([Ax2_1, Bx, C], name='y1')

# y = Ax^2 + Bx^2
Ax2_2 = tf.multiply(A, tf.pow(x,2),name='Ax2_2')
Bx2 = tf.multiply(B, tf.pow(x,2),name='Bx2')
y2 = tf.add_n([Ax2_2,Bx2],name='y2')
y = y1 + y2

with tf.Session() as sess:
	print(sess.run(y, feed_dict={x:[10]}))

	writer = tf.summary.FileWriter('./named_scope', sess.graph)
	writer.close()

#%%
	
import tensorflow as tf
A = tf.constant([4], tf.int32, name='A')
B = tf.constant([4], tf.int32, name='B')
C = tf.constant([4], tf.int32, name='C')
x = tf.placeholder(tf.int32, name='x')

# y = Ax^2 + Bx + C
with tf.name_scope("Equation1"):
	Ax2_1 = tf.multiply(A, tf.pow(x,2), name="Ax2_1")
	Bx = tf.multiply(A,x, name="Bx")
	y1 = tf.add_n([Ax2_1, Bx, C], name='y1')

# y = Ax^2 + Bx^2
with tf.name_scope("Equation2"):
	Ax2_2 = tf.multiply(A, tf.pow(x,2),name='Ax2_2')
	Bx2 = tf.multiply(B, tf.pow(x,2),name='Bx2')
	y2 = tf.add_n([Ax2_2,Bx2],name='y2')

with tf.name_scope("final_sum"):
	y = y1 + y2

with tf.Session() as sess:
	print(sess.run(y, feed_dict={x:[10]}))

writer = tf.summary.FileWriter('./named_scope',sess.graph)
writer.close()

#%%

''' tf.lin_space(start, stop, num, name=None) '''
QQQ = tf.lin_space(10.0, 13.0, 20)     # result is [10. 11. 12. 13.]

with tf.Session() as sess:
	print(sess.run(QQQ))
	
''' tf.range(start, limit=None, delta=1, dtype=None, name='range') '''
YYY = tf.range(3, 18, 3, name='MyRange')              # result is [3 6 9 12 15]
ZZZ = tf.range(5)                     # result is [0 1 2 3 4]

with tf.Session() as sess:
	print(sess.run(YYY))
	print(sess.run(ZZZ))
