from header import *

X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=())
Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])

sampled_z, mean, std = encoder(X, keep_prob)
decoded = decoder(sampled_z, keep_prob)
decoded_unreshaped = tf.reshape(decoded, [-1, 28*28])

sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint("./model/"))

path = os.getcwd() + '/MNIST_all'
mnist_list = [name for name in glob.glob(os.path.join(path, '*.jpg'))]
mnist_dataset = dataset(mnist_list)

batch_size = 64
loss_values = []


for i in range(20):

	batch = [np.array(Image.open(name))/255 for name in mnist_dataset.next_batch(batch_size)]

	dec = sess.run(decoded, feed_dict={X: batch, Y: batch, keep_prob: 1.0})
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(np.reshape(batch[2], [28, 28]), cmap='gray')
	ax2.imshow(dec[2], cmap='gray')
	ax1.axis('off')
	ax2.axis('off')
	plt.savefig("comparison/{0}.jpg".format(i))
	plt.close()