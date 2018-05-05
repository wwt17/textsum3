from vocab_utils import Vocab as EntVocab
from SentenceMatchModelGraph import SentenceMatchModelGraph

if __name__ == "__main__":
    config_path = ""
    word_vec_path = ""
    ent_word_vocab = EntVocab(word_vec_path, fileformat='txt3')

    global_step = tf.train.get_global_step()

    init_scale = 0.01
    with tf.variable_scope("Model",
        initializer=tf.random_uniform_initializer(-init_scale, init_scale)):
        config_FLAGS = namespace_utils.load_namespace(config_path)
        entailment_model = SentenceMatchModelGraph(
            3, word_vocab=ent_word_vocab, is_training=True,
            options=config_FLAGS, global_step=global_step)
        disc_result = entailment_model() #TODO
