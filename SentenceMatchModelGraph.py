import tensorflow as tf
import layer_utils
import match_utils

import texar as tx

class SentenceMatchModelGraph(tx.modules.ModuleBase):
    def __init__(self, num_classes, word_vocab=None, is_training=True, options=None, global_step=None, hparams=None):
        tx.modules.ModuleBase.__init__(self, hparams)
        self.options = options
        self.num_classes = num_classes
        self.word_vocab = word_vocab
        self.is_training = is_training
        self.global_step = global_step

    def _build(self, in_passage_words, passage_lengths, in_question_words_soft, question_lengths, truth):
        """ truth: a int in [0 .. num_classes] indicating entailment
        """
        num_classes = self.num_classes
        word_vocab = self.word_vocab
        is_training = self.is_training
        global_step = self.global_step
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

            #in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, in_question_words_soft) # [batch_size, question_len, word_dim]
            in_question_word_repres = tx.utils.soft_sequence_embedding(
                self.word_embedding, in_question_words_soft)
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(in_question_words_soft)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim
            
        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))

        mask = tf.sequence_mask(passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)

        # in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
        # in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(mask, axis=-1))

        # ========Bilateral Matching=====
        (match_representation, match_dim) = match_utils.bilateral_match_func(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, input_dim, is_training, options=options)

        #========Prediction Layer=========
        # match_dim = 4 * self.options.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        if is_training: logits = tf.nn.dropout(logits, (1 - options.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.prob = tf.nn.softmax(logits)
        
        gold_matrix = tf.one_hot(truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

        correct = tf.nn.in_top_k(logits, truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.argmax(self.prob, 1)

        if is_training:
            tvars = tf.trainable_variables()
            if self.options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + self.options.lambda_l2 * l2_loss

            if self.options.optimize_type == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
            elif self.options.optimize_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)

            grads = layer_utils.compute_gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            if self.options.with_moving_average:
                # Track the moving averages of all trainable variables.
                MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
                variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                train_ops = [self.train_op, variables_averages_op]
                self.train_op = tf.group(*train_ops)
        
        return {
            "logits": logits,
            "prob": self.prob,
            "loss": self.loss,
            "correct": correct,
            "eval_correct": self.eval_correct,
            "predictions": self.predictions,
        }
