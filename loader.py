from header import *

sampled_z = tf.placeholder(dtype=tf.float32, shape=[1, latent_dim])
decoded = decoder(sampled_z, keep_prob)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint("./model/"))

for i in range(20):

	random_latent = np.random.normal(size=(1, latent_dim))

	img = sess.run(decoded, feed_dict={sampled_z: random_latent, keep_prob: 1.0})
	img = np.reshape(img, [28, 28])

	plt.imshow(img, cmap='gray')
	plt.axis("off")
	plt.savefig("generated/img_gen{0}.jpg".format(i))
	plt.close()