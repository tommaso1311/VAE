from header import *

Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])

sampled_z, mean, std = encoder(X, keep_prob)
decoded = decoder(sampled_z, keep_prob)
decoded_unreshaped = tf.reshape(decoded, [-1, 28*28])

img_loss = tf.reduce_sum(tf.squared_difference(decoded_unreshaped, Y_flat), axis=1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) - tf.exp(2.0 * std), axis=1)
loss = tf.reduce_mean(img_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

path = os.getcwd() + '/MNIST_all'
mnist_list = [name for name in glob.glob(os.path.join(path, '*.jpg'))]
mnist_dataset = dataset(mnist_list)

batch_size = 64
loss_values = []

epochs = 3001

for i in range(epochs):

	batch = [np.array(Image.open(name))/255 for name in mnist_dataset.next_batch(batch_size)]

	sess.run(optimizer, feed_dict={X: batch, Y: batch, keep_prob: 0.8})

	if not i % 100:

		ls, dec, img_ls, dec_ls, mn, sd = sess.run([loss, decoded, img_loss, latent_loss, mean, std], feed_dict={X: batch, Y: batch, keep_prob: 1.0})
		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
		ax2.imshow(dec[0], cmap='gray')
		ax1.axis('off')
		ax2.axis('off')
		plt.savefig("images/{0}.jpg".format(i))
		plt.close()
		loss_values.append(ls)

		print(i, ls, np.mean(img_ls), np.mean(dec_ls))

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(0, epochs, 100), loss_values, 'r')
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
seaborn.despine(fig, top=True, right=True, trim=True, offset=10)
plt.savefig("images/loss.jpg")
plt.close()

saver = tf.train.Saver()
saver.save(sess, "./model//model.ckpt", global_step=epochs)