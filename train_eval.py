import datetime
import logging
import os

import numpy as np
import time
import torch
import torch.nn

from model import MRAN
# DistMult, ComplEx, ConvE, MRAN
from load_data import Data
from radam import RAdam


class RunModel:

    def __init__(self, data: Data, modelname, optimizer_method="Adam", learning_rate=0.001, ent_vec_dim=200,
                 rel_vec_dim=200, num_iterations=100, batch_size=128, decay_rate=0., cuda=False, input_dropout=0.,
                 hidden_dropout=0., feature_map_dropout=0., in_channels=1, out_channels=32, filt_h=3, filt_w=3,
                 label_smoothing=0., num_to_eval=10, get_best_results=True, get_complex_results=True,
                 regular_method='L2', regular_rate=0.0001):

        self.cuda = cuda
        self.data = data
        self.model_name = modelname
        self.optimizer_method = optimizer_method
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.num_to_eval = num_to_eval
        self.get_best_results = get_best_results
        self.get_complex_results = get_complex_results
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_height": filt_h, "filt_width": filt_w}
        self.best_hits_10, self.best_hits_3, self.best_hits_1, self.best_mr, self.best_mrr = 0, 0, 0, 1e7, 0
        if self.get_complex_results:
            self.get_best_results = False

        # 记录训练参数
        now_time = datetime.datetime.now().strftime('_%m-%d-%H-%M-%S')
        logging.basicConfig(filename=os.path.join(os.getcwd(), 'results/%s.txt' % (
                self.model_name + "_" + self.data.data_name + str(now_time))),
                            level=logging.NOTSET, format='%(message)s')
        logging.info("data name: " + self.data.data_name)
        logging.info("model name: " + self.model_name)
        logging.info("learning rate: " + str(self.learning_rate))
        logging.info("entity dimension: " + str(self.ent_vec_dim))
        logging.info("relation dimension: " + str(self.rel_vec_dim))
        logging.info("batch size: " + str(self.batch_size))
        logging.info("decay rate: " + str(self.decay_rate))
        logging.info("label smoothing: " + str(self.label_smoothing))
        logging.info("convolution parameters: " + str(self.kwargs) + '\n')

        logging.info('Epoch' + '\t' + 'MR' + '\t' + 'MRR' + '\t' + 'Hits1' + '\t' + 'Hits3' + '\t' + 'Hits10' + '\t' +
                     'Best_MR' + '\t' + 'Best_MRR' + '\t' + 'Best_Hits1' + '\t' + 'Best_Hits3' + '\t' + 'Best_Hits10')

        self.regular_method = regular_method
        self.regular_rate = regular_rate

    def regular_loss(self, param):
        loss = 0
        size = param.size()

        if self.regular_method.lower() == "l1":
            loss = torch.norm(param, p=1)
        if self.regular_method.lower() == "l2":
            loss = torch.norm(param, p=2) ** 2
        if self.regular_method.lower() == "huber":
            huber = torch.nn.SmoothL1Loss(reduction='sum')
            loss = huber(param, torch.tensor(0).repeat(size).float())
            # param = abs(param.reshape(-1))
            # for i in param:
            #     if i <= self.huber_delta:
            #         loss += i ** 2
            #     else:
            #         loss += (2 * i - self.huber_delta) * self.huber_delta
        return loss

    def evaluate(self, model, eval_data, iteration):
        rank_filt = []
        hits_filt = []
        for i in range(10):
            hits_filt.append([])

        for batch_data, batch_num in self.data.get_batch_eval_data(self.batch_size, eval_data):
            head_id = batch_data[:, 0]
            rela_id = batch_data[:, 1]
            tail_id = batch_data[:, 2]
            if self.cuda:
                head_id = head_id.cuda()
                rela_id = rela_id.cuda()
                tail_id = tail_id.cuda()
            pred = model.forward(head_id, rela_id)

            # get filter rank
            for i in range(batch_num):
                filt = self.data.all_hr_dict[(head_id[i].item(), rela_id[i].item())]
                target_value = pred[i, tail_id[i]].item()
                pred[i, filt] = 0.0
                pred[i, tail_id[i]] = target_value

            _, filt_sort_id = torch.topk(pred, k=self.data.entities_num)
            filt_sort_id = filt_sort_id.cpu().numpy()

            for i in range(batch_num):
                rank_f = np.where(filt_sort_id[i] == tail_id[i].item())[0][0]
                rank_filt.append(rank_f + 1)

                for hits_level in range(10):
                    if rank_f <= hits_level:
                        hits_filt[hits_level].append(1.0)
                    else:
                        hits_filt[hits_level].append(0.0)

        filt_hits_10, filt_hits_3, filt_hits_1 = np.mean(hits_filt[9]), np.mean(hits_filt[2]), np.mean(hits_filt[0])
        filt_mr, filt_mrr = np.mean(rank_filt), np.mean(1. / np.array(rank_filt))

        # return the best results.
        if self.best_hits_10 < filt_hits_10:
            self.best_hits_10 = filt_hits_10
        if self.best_hits_3 < filt_hits_3:
            self.best_hits_3 = filt_hits_3
        if self.best_hits_1 < filt_hits_1:
            self.best_hits_1 = filt_hits_1
        if self.best_mr > filt_mr:
            self.best_mr = filt_mr
        if self.best_mrr < filt_mrr:
            self.best_mrr = filt_mrr

        if self.get_best_results:
            print('----- [%s]: [%s] results -----' % (self.model_name, self.data.data_name))
            print('Hits  @10: %.3f, Best  Hits @10: %.3f' % (filt_hits_10, self.best_hits_10))
            print('Hits   @3: %.3f, Best  Hits  @3: %.3f' % (filt_hits_3, self.best_hits_3))
            print('Hits   @1: %.3f, Best  Hits  @1: %.3f' % (filt_hits_1, self.best_hits_1))
            print('MR : %.3f, Best MR : %.3f' % (filt_mr, self.best_mr))
            print('MRR: %.3f, Best MRR: %.3f' % (filt_mrr, self.best_mrr))

            # logging.info('----- [%s]: [%s] results -----' % (self.model_name, self.data.data_name))
            # logging.info('Hits  @10: %.3f, Best  Hits @10: %.3f' % (filt_hits_10, self.best_hits_10))
            # logging.info('Hits   @3: %.3f, Best  Hits  @3: %.3f' % (filt_hits_3, self.best_hits_3))
            # logging.info('Hits   @1: %.3f, Best  Hits  @1: %.3f' % (filt_hits_1, self.best_hits_1))
            # logging.info('MR : %.3f, Best MR : %.3f' % (filt_mr, self.best_mr))
            # logging.info('MRR: %.3f, Best MRR: %.3f' % (filt_mrr, self.best_mrr))
            logging.info(str(iteration + 1) + '\t' + str('%.1f' % (filt_mr)) + '\t' + str('%.3f' % (filt_mrr)) + '\t' +
                         str('%.3f' % (filt_hits_1)) + '\t' + str('%.3f' % (filt_hits_3)) + '\t' +
                         str('%.3f' % (filt_hits_10)) + '\t' +
                         str('%.1f' % (self.best_mr)) + '\t' + str('%.3f' % (self.best_mrr)) + '\t' +
                         str('%.3f' % (self.best_hits_1)) + '\t' + str('%.3f' % (self.best_hits_3)) + '\t' +
                         str('%.3f' % (self.best_hits_10)))

        return filt_hits_10, filt_hits_3, filt_hits_1, filt_mr, filt_mrr

    def train_and_eval(self):
        model = MRAN(self.data, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        # print([param.numel() for param in model.parameters()])
        # print([(name, param) for name, param in model.named_parameters()])

        if self.cuda:
            model.cuda()
        model.init()
        if self.optimizer_method.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_method.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_method.lower() == 'radam':
            optimizer = RAdam(model.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.decay_rate)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=self.decay_rate)
            # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.decay_rate, patience=3)

        print('----- Start training -----')
        for iteration in range(self.num_iterations):
            begin_time = time.time()
            model.train()
            epoch_loss = 0
            for batch_data, batch_target in self.data.get_batch_train_data(self.batch_size):
                # Clears the gradients of all optimized
                optimizer.zero_grad()

                head_id = batch_data[:, 0]
                rela_id = batch_data[:, 1]
                if self.cuda:
                    head_id = head_id.cuda()
                    rela_id = rela_id.cuda()
                    batch_target = batch_target.cuda()

                pred = model.forward(head_id, rela_id)

                # regularization
                regular_loss = 0
                # for name, param in model.named_parameters():
                #     if 'entity_embedding' in name:
                #         regular_loss += self.regular_loss(param)
                #     if 'relation_embedding' in name:
                #         regular_loss += self.regular_loss(param)
                #     if 'filter' in name:
                #         regular_loss += self.regular_loss(param)
                for name, param in model.named_parameters():
                    # if 'weight' in name:
                    regular_loss += self.regular_loss(param)

                if self.label_smoothing:
                    batch_target = ((1.0 - self.label_smoothing) * batch_target) + (1.0 / batch_target.size(1))
                #batch_loss = model.loss(pred, batch_target) + (self.regular_rate / 2) * regular_loss
                batch_loss = model.loss(pred, batch_target)
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

            if self.decay_rate:
                scheduler.step(epoch_loss)
            end_time = time.time()
            print('Iteration:' + str(iteration) + '   ' + 'epoch loss: {:.5f}'.format(epoch_loss) +
                  '   ' + 'time cost: {:.3f}'.format(end_time - begin_time))

            # logging.info('Iteration:' + str(iteration) + '   ' + 'mean loss: {:.5f}'.format(epoch_loss) +
            #              '   ' + 'time cost: {:.3f}'.format(time.time() - begin_time))

            model.eval()
            with torch.no_grad():
                if (iteration + 1) % self.num_to_eval == 0:
                    if self.get_complex_results:
                        print('----- [%s]: 1-1 results -----' % self.model_name)
                        logging.info('----- [%s]: 1-1 results -----' % self.model_name)
                        self.get_complex_result(model, self.data.O_O_hr_t_id, self.data.O_O_tr_h_id)

                        # 1-n
                        print('----- [%s]: 1-n results -----' % self.model_name)
                        logging.info('----- [%s]: 1-n results -----' % self.model_name)
                        self.get_complex_result(model, self.data.O_N_hr_t_id, self.data.O_N_tr_h_id)

                        # n-1
                        print('----- [%s]: n-1 results -----' % self.model_name)
                        logging.info('----- [%s]: n-1 results -----' % self.model_name)
                        self.get_complex_result(model, self.data.N_O_hr_t_id, self.data.N_O_tr_h_id)

                        # n-n
                        print('----- [%s]: n-n results -----' % self.model_name)
                        logging.info('----- [%s]: n-n results -----' % self.model_name)
                        self.get_complex_result(model, self.data.N_N_hr_t_id, self.data.N_N_tr_h_id)
                    else:
                        # print("----- Valid_data evaluation-----")
                        # self.evaluate(model, self.data.valid_data_id, iteration)
                        print('----- Test_data evaluation -----')
                        self.evaluate(model, self.data.test_data_id, iteration)

                    # evaluate relations category for WN18
                    # else:
                    #     print('----- Test_data evaluation -----')
                    #     # 1-1
                    #     self.evaluate(model, self.data.rela13, iteration)
                    #     self.evaluate(model, self.data.rela17, iteration)
                    #
                    #     # 1-N
                    #     self.evaluate(model, self.data.rela2, iteration)
                    #     self.evaluate(model, self.data.rela4, iteration)
                    #     self.evaluate(model, self.data.rela6, iteration)
                    #     self.evaluate(model, self.data.rela8, iteration)
                    #     self.evaluate(model, self.data.rela9, iteration)
                    #     self.evaluate(model, self.data.rela10, iteration)
                    #     self.evaluate(model, self.data.rela11, iteration)
                    #
                    #     # N-1
                    #     self.evaluate(model, self.data.rela3, iteration)
                    #     self.evaluate(model, self.data.rela5, iteration)
                    #     self.evaluate(model, self.data.rela7, iteration)
                    #     self.evaluate(model, self.data.rela12, iteration)
                    #     self.evaluate(model, self.data.rela14, iteration)
                    #     self.evaluate(model, self.data.rela15, iteration)
                    #     self.evaluate(model, self.data.rela16, iteration)
                    #
                    #     # N-N
                    #     self.evaluate(model, self.data.rela0, iteration)
                    #     self.evaluate(model, self.data.rela1, iteration)

    def get_complex_result(self, model, hr_t_data, tr_h_data):
        hr_t_hits_10, hr_t_hits_3, hr_t_hits_1, hr_t_mr, hr_t_mrr = self.evaluate(model, hr_t_data, iteration=2500)
        tr_h_hits_10, tr_h_hits_3, tr_h_hits_1, tr_h_mr, tr_h_mrr = self.evaluate(model, tr_h_data, iteration=2500)
        print('Pred Head [Hits @10]: %.3f, Pred Tail [Hits @10]: %.3f' % (tr_h_hits_10, hr_t_hits_10))
        print('Pred Head [Hits @3 ]: %.3f, Pred Tail [Hits @3 ]: %.3f' % (tr_h_hits_3, hr_t_hits_3))
        print('Pred Head [Hits @1 ]: %.3f, Pred Tail [Hits @1 ]: %.3f' % (tr_h_hits_1, hr_t_hits_1))
        print('Pred Head [ MR  ]: %.1f, Pred Tail [ MR  ]: %.1f' % (tr_h_mr, hr_t_mr))
        print('Pred Head [ MRR ]: %.3f, Pred Tail [ MRR ]: %.3f' % (tr_h_mrr, hr_t_mrr))
        logging.info('Pred Head [Hits @10]: %.3f, Pred Tail [Hits @10]: %.3f' % (tr_h_hits_10, hr_t_hits_10))
        logging.info('Pred Head [Hits @3 ]: %.3f, Pred Tail [Hits @3 ]: %.3f' % (tr_h_hits_3, hr_t_hits_3))
        logging.info('Pred Head [Hits @1 ]: %.3f, Pred Tail [Hits @1 ]: %.3f' % (tr_h_hits_1, hr_t_hits_1))
        logging.info('Pred Head [ MR  ]: %.1f, Pred Tail [ MR  ]: %.1f' % (tr_h_mr, hr_t_mr))
        logging.info('Pred Head [ MRR ]: %.3f, Pred Tail [ MRR ]: %.3f' % (tr_h_mrr, hr_t_mrr))
