import tensorflow as tf
import os
from vocab_utils import Vocab as EntVocab
import namespace_utils
from SentenceMatchModelGraph import SentenceMatchModelGraph

if __name__ == "__main__":
    config_path = "./configs/snli.sample.config"
    config_FLAGS = namespace_utils.load_namespace(config_path)
    config_FLAGS.__dict__["in_format"] = 'tsv'
    word_vec_path = config_FLAGS.word_vec_path
    log_dir = config_FLAGS.model_dir
    path_prefix = os.path.join(log_dir, "SentenceMatch.{}".format(config_FLAGS.suffix))

    ent_word_vocab = EntVocab(word_vec_path, fileformat='txt3')
    print("word_vocab shape is {}".format(ent_word_vocab.word_vecs.shape))

    best_path = path_prefix + ".best.model"
    label_path = path_prefix + ".label_vocab"
    print("best_path: {}".format(best_path))

    if os.path.exists(best_path + ".index"):
        print("Loading label vocab")
        label_vocab = EntVocab(label_path, fileformat='txt2')
    else:
        raise Exception("no pretrained model")

    num_classes = label_vocab.size()
    print("Number of labels: {}".format(num_classes))

    global_step = tf.train.get_global_step()

    init_scale = 0.01
    with tf.variable_scope("Model",
        initializer=tf.random_uniform_initializer(-init_scale, init_scale)):
        config_FLAGS = namespace_utils.load_namespace(config_path)
        entailment_model = SentenceMatchModelGraph(
            3, word_vocab=ent_word_vocab, is_training=True,
            options=config_FLAGS, global_step=global_step)
        disc_result = entailment_model(
            tf.placeholder(tf.int32, [None, None]),
            tf.placeholder(tf.int32, [None]),
            tf.placeholder(tf.int32, [None, None]),
            tf.placeholder(tf.int32, [None]),
            tf.placeholder(tf.int32, [None]))

    initializer = tf.global_variables_initializer()
    vars_ = {}
    for var in tf.global_variables():
        if "word_embedding" in var.name:
            continue
        vars_[var.name.split(":")[0]] = var
    for key, value in vars_.items():
        print('key={} value={}'.format(key, value))
    saver = tf.train.Saver(vars_)

    with tf.Session() as sess:
        sess.run(initializer)
        print("Restoring model from {}".format(best_path))
        saver.restore(sess, best_path)
        print("DONE!")
