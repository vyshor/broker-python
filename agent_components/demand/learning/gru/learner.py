import logging
import time

from keras.layers import CuDNNGRU, regularizers
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.utils import Sequence

import util.config as cfg
from agent_components.demand.learning.preprocessing import DemandCustomerSequence, drain_generator
from util.strings import EPOCH_NO_WITH_LOSS, GAME_NUMBER_X, FIT_WITH_GENERATOR
from util.learning_utils import StatefileCapableLearner, ModelWriter, TbWriterHelper, GamesIterator, get_callbacks

log = logging.getLogger(__name__)


class Learner(StatefileCapableLearner):
    """This class' "run" method is always called by the `main.py` Click CLI based script. So when implementing new learning models,
    you need this"""
    def run_with_demand_files(self):
        mdl = self.get_model()

        #get some helpers
        mw = ModelWriter(self.model_name)
        tb_writer = TbWriterHelper(self.model_name)
        # iterate over games (holding customer lists)
        log.info("running full learning for demand:")
        for g_number, game in enumerate(GamesIterator('demand')):
            log.info(GAME_NUMBER_X.format(g_number))
            train, validation = self.split_game(game)
            generator = DemandCustomerSequence(cfg.DEMAND_FORECAST_DISTANCE, train[0], train[1])
            validation_sequences = drain_generator(DemandCustomerSequence(cfg.DEMAND_FORECAST_DISTANCE, validation[0], validation[1]), 10)
            self.run_epoch(mdl, generator, validation_sequences, g_number, tb_writer)
            #self.fit_with_generator(mdl, generator, self.keras_callbacks, validation_set=validation_sequences)
            mw.write_model(mdl)


    def get_model(self):
        model = Sequential()

        #input layer
        model.add(CuDNNGRU (input_shape=(int(cfg.DEMAND_SEQUENCE_LENGTH / cfg.DEMAND_SAMPLING_RATE),
                                         cfg.DEMAND_DATAPOINTS_PER_TS),
                            units=cfg.DEMAND_DATAPOINTS_PER_TS,
                            kernel_regularizer=regularizers.l1(0.01),
                            return_sequences=True,
                            ))
        model.add(CuDNNGRU(units=200,
                           return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNGRU(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=100))
        model.add(Dense(units=1))
        model.add(Activation('linear'))

        start = time.time()
        optimizr = rmsprop(lr=0.01)
        model.compile(loss='mae', optimizer=optimizr)
        log.info('compilation time : {}'.format(time.time() - start))
        return model



    def fit_with_generator(self, model, train_generator: Sequence, callbacks, validation_set=None):
        log.info(FIT_WITH_GENERATOR.format(len(train_generator), len(validation_set[1])))
        return model.fit_generator(train_generator,
                            steps_per_epoch=None,  #a generator size (aka one customer) is an epoch
                            #steps_per_epoch=200,  #a generator size (aka one customer) is an epoch
                            epochs=1,
                            verbose=1,  #progress bar, 2 = line per epoch
                            callbacks=callbacks,
                            validation_data=validation_set,
                            #validation_steps=None,
                            class_weight=None,
                            max_queue_size=10,
                            workers=8,
                            use_multiprocessing=True,
                            shuffle=False,
                            initial_epoch=0)

    def run_epoch(self, mdl, generator, validation, epoch_n, tb_writer):
        loss = -1
        for i in range(len(generator)):
            x, y, = generator.__getitem__(i)
            loss = mdl.train_on_batch(x, y)
            if i % 100:
                tb_writer.write_train_loss(loss)
        test_loss = mdl.test_on_batch(validation[0], validation[1])
        log.info(EPOCH_NO_WITH_LOSS.format(epoch_n, test_loss))
        tb_writer.write_test_loss(test_loss)
