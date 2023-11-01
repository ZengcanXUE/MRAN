from collections import defaultdict

import numpy as np
import torch


class Data:

    def __init__(self, data_dir, reverse=False):
        self.data_name = data_dir[5:-1]
        # 获取训练、验证、测试数据集
        # [['/m/03p41', '/medicine/disease/medical_specialties', '/m/03tp4'], ...]
        self.train_data, self.train_data_num, self.train_hrt = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data, self.valid_data_num, self.valid_hrt = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data, self.test_data_num, self.test_hrt = self.load_data(data_dir, "test", reverse=reverse)
        self.all_data = self.train_data + self.valid_data + self.test_data
        self.all_hrt = self.train_hrt + self.valid_hrt + self.test_hrt

        # self.train_data.extend(self.valid_data)
        # self.valid_data = []

        # 获取实体, 关系数据: entities: ['/m/010016', '/m/0100mt', '/m/0102t4', ...]
        # 建立实体及关系dict，获取索引: entities_id: {'/m/027rn':0, ...}
        self.entities, self.entities_id, self.entities_num = self.get_entities(self.all_data)
        self.relations, self.relations_id, self.relations_num = self.get_relations(self.all_data)

        # 三元组实体及关系索引的list: [(2050, 20, 5037), ...]
        self.train_data_id = self.data_id(self.train_data)
        self.valid_data_id = self.data_id(self.valid_data)
        self.test_data_id = self.data_id(self.test_data)
        self.all_data_id = self.data_id(self.all_data)

        # dict: {(2050, 20): [5037, 4545...], (8810, 119): [4564,..], ...}
        # list: [(8500, 730), (5475, 1041), (11650, 99),...]
        self.train_hr_dict = self.get_hr_dict(self.train_data_id)
        self.train_hr_list = list(self.train_hr_dict.keys())
        self.train_hr_list_num = len(self.train_hr_list)
        self.all_hr_dict = self.get_hr_dict(self.all_data_id)

        # list: [(8500, 730, 5475), (5475, 1041, 99), ...]
        self.O_O_hr_t_id, self.O_N_hr_t_id, self.N_O_hr_t_id, self.N_N_hr_t_id, \
        self.O_O_tr_h_id, self.O_N_tr_h_id, self.N_O_tr_h_id, self.N_N_tr_h_id = self.get_complex_triple()

        print('数据集: {}'.format(self.data_name))
        print('实体数: {}'.format(self.entities_num), '关系数: {}'.format(self.relations_num))
        print('训练集: {}'.format(self.train_data_num),
              '验证集: {}'.format(self.valid_data_num),
              '测试集: {}'.format(self.test_data_num))

        # relations category for WN18
        # 1-1:  ('_similar_to': 13), ('_verb_group': 17)
        self.rela13 = [i for i in self.test_data_id if i[1] == 26] + [i for i in self.test_data_id if i[1] == 27]
        self.rela17 = [i for i in self.test_data_id if i[1] == 34] + [i for i in self.test_data_id if i[1] == 35]

        # 1-N:  ('_has_part': 2), ('_hyponym': 4), ('_instance_hyponym': 6), ('_member_meronym': 8),
        #       ('_member_of_domain_region': 9), ('_member_of_domain_topic': 10), ('_member_of_domain_usage': 11),
        self.rela2 = [i for i in self.test_data_id if i[1] == 4] + [i for i in self.test_data_id if i[1] == 5]
        self.rela4 = [i for i in self.test_data_id if i[1] == 8] + [i for i in self.test_data_id if i[1] == 9]
        self.rela6 = [i for i in self.test_data_id if i[1] == 12] + [i for i in self.test_data_id if i[1] == 13]
        self.rela8 = [i for i in self.test_data_id if i[1] == 16] + [i for i in self.test_data_id if i[1] == 17]
        self.rela9 = [i for i in self.test_data_id if i[1] == 18] + [i for i in self.test_data_id if i[1] == 19]
        self.rela10 = [i for i in self.test_data_id if i[1] == 20] + [i for i in self.test_data_id if i[1] == 21]
        self.rela11 = [i for i in self.test_data_id if i[1] == 22] + [i for i in self.test_data_id if i[1] == 23]

        # N-1:  ('_hypernym': 3), ('_instance_hypernym': 5), ('_member_holonym': 7), ('_part_of': 12),
        #       ('_synset_domain_region_of': 14), ('_synset_domain_topic_of': 15), ('_synset_domain_usage_of': 16)
        self.rela3 = [i for i in self.test_data_id if i[1] == 6] + [i for i in self.test_data_id if i[1] == 7]
        self.rela5 = [i for i in self.test_data_id if i[1] == 10] + [i for i in self.test_data_id if i[1] == 11]
        self.rela7 = [i for i in self.test_data_id if i[1] == 14] + [i for i in self.test_data_id if i[1] == 15]
        self.rela12 = [i for i in self.test_data_id if i[1] == 24] + [i for i in self.test_data_id if i[1] == 25]
        self.rela14 = [i for i in self.test_data_id if i[1] == 28] + [i for i in self.test_data_id if i[1] == 29]
        self.rela15 = [i for i in self.test_data_id if i[1] == 30] + [i for i in self.test_data_id if i[1] == 31]
        self.rela16 = [i for i in self.test_data_id if i[1] == 32] + [i for i in self.test_data_id if i[1] == 33]

        # N-N   ('_also_see': 0), ('_derivationally_related_form': 1)
        self.rela0 = [i for i in self.test_data_id if i[1] == 0] + [i for i in self.test_data_id if i[1] == 1]
        self.rela1 = [i for i in self.test_data_id if i[1] == 2] + [i for i in self.test_data_id if i[1] == 3]
        # print(self.rela0)

    @staticmethod
    def get_hr_dict(data_id):
        """
        获取同一头实体与关系对应的尾实体的dict
        :param data_id: 用id表示的三元组list
        :return: er_vocab {(2050, 20): [5037,4545...], (8810, 119): [4564,..], ...}
        """
        hr_dict = defaultdict(list)
        for triple in data_id:
            hr_dict[(triple[0], triple[1])].append(triple[2])
        return hr_dict

    # @staticmethod
    # def get_tr_dict(data_id):
    #     """
    #     获取同一头实体与关系对应的尾实体的dict
    #     :param data_id: 用id表示的三元组list
    #     :return: er_vocab {(2050, 20): [5037,4545...], (8810, 119): [4564,..], ...}
    #     """
    #     tr_dict = defaultdict(list)
    #     for triple in data_id:
    #         tr_dict[(triple[2], triple[1])].append(triple[0])
    #     return tr_dict

    def get_batch_train_data(self, batch_size):
        """
        :param batch_size: batch_size
        :return: batch_data：nparray, batch_size大小的train_hr_list
                 [[8500, 730], [5475, 1041], [11650, 99],...]
                 batch_targets：浮点型tensor， 对应尾实体序号位置赋值为1
        """
        start = 0
        np.random.shuffle(self.train_hr_list)
        while start < self.train_hr_list_num:
            end = min(start + batch_size, self.train_hr_list_num)
            batch_data = self.train_hr_list[start:end]

            batch_target = np.zeros((len(batch_data), self.entities_num))
            for index, hr_pair in enumerate(batch_data):
                batch_target[index, self.train_hr_dict[hr_pair]] = 1.0

            batch_data = torch.tensor(batch_data)
            batch_target = torch.FloatTensor(batch_target)

            start = end
            yield batch_data, batch_target

    @staticmethod
    def get_batch_eval_data(batch_size, eval_data):
        """
        :param batch_size: batch_size
        :param eval_data: 测试数据
        :return:
        """
        eval_data_num = len(eval_data)
        start = 0
        while start < eval_data_num:
            end = min(start + batch_size, eval_data_num)

            batch_data = eval_data[start:end]
            batch_num = len(batch_data)
            batch_data = torch.tensor(batch_data)

            start = end
            yield batch_data, batch_num

    # 获取三元组的id表示
    def data_id(self, data):
        data_num = len(data)
        data_id = [(self.entities_id[data[i][0]],
                    self.relations_id[data[i][1]],
                    self.entities_id[data[i][2]])
                   for i in range(data_num)]
        return data_id

    # 获取三元组数据
    @staticmethod
    def load_data(data_dir, data_type, reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            # , encoding='utf-8'
            # strip()：方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            # split("\r\n")：'\r'是回车，'\n'是换行，前者使光标到行首，后者使光标下移一格。
            # 字符串
            # list列表
            hrt = f.read().strip().split("\n")
            hrt = [i.split() for i in hrt]
            trh = []
            if reverse:
                trh = [[i[2], i[1] + "_reverse", i[0]] for i in hrt]
            data = hrt + trh
            data_num = len(data)
            f.close()
        return data, data_num, hrt

    # 获取实体数据
    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        entities_num = len(entities)
        entities_id = {entities[i]: i for i in range(entities_num)}
        return entities, entities_id, entities_num

    # 获取关系数据
    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        relations_num = len(relations)
        relations_id = {relations[i]: i for i in range(relations_num)}
        return relations, relations_id, relations_num

    def get_complex_triple(self):
        """
        获取1-1, 1-n, n-1, n-n三元组id表示
        :return: O_O_triple, O_N_triple, N_O_triple, N_N_triple
        """
        # dict:{0:[1 ,2 ,3, ...], 1..}
        relation_to_head = defaultdict(set)
        relation_to_tail = defaultdict(set)
        relation_to_trip = defaultdict(int)

        for h, r, t in self.all_hrt:
            relation_to_head[r].add(h)
            relation_to_tail[r].add(t)
            relation_to_trip[r] += 1

        # hpt等于关于r的所有三元组除以关于r的所有尾实体
        # tph等于关于r的所有三元组除以关于r的所有头实体
        relation_hpt = {r: relation_to_trip[r] / len(relation_to_tail[r]) for h, r, t in self.all_hrt}
        relation_tph = {r: relation_to_trip[r] / len(relation_to_head[r]) for h, r, t in self.all_hrt}

        O_O_triple_hr_t, O_N_triple_hr_t, N_O_triple_hr_t, N_N_triple_hr_t = [], [], [], []
        # O_O_triple_tr_h, O_N_triple_tr_h, N_O_triple_tr_h, N_N_triple_tr_h = [], [], [], []
        for h, r, t in self.test_hrt:
            if relation_hpt[r] <= 1.5 and relation_tph[r] <= 1.5:
                O_O_triple_hr_t.append([h, r, t])
                # O_O_triple_tr_h = [[i[2], i[1] + "_reverse", i[0]] for i in O_O_triple_hr_t]
            elif relation_hpt[r] <= 1.5 and relation_tph[r] > 1.5:
                O_N_triple_hr_t.append([h, r, t])
                # O_N_triple_tr_h = [[i[2], i[1] + "_reverse", i[0]] for i in O_N_triple_hr_t]
            elif relation_hpt[r] > 1.5 and relation_tph[r] <= 1.5:
                N_O_triple_hr_t.append([h, r, t])
                # N_O_triple_tr_h = [[i[2], i[1] + "_reverse", i[0]] for i in N_O_triple_hr_t]
            elif relation_hpt[r] > 1.5 and relation_tph[r] > 1.5:
                N_N_triple_hr_t.append([h, r, t])
                # N_N_triple_tr_h = [[i[2], i[1] + "_reverse", i[0]] for i in N_N_triple_hr_t]

        O_O_hr_t_id = self.data_id(O_O_triple_hr_t)
        O_O_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in O_O_hr_t_id]

        O_N_hr_t_id = self.data_id(O_N_triple_hr_t)
        O_N_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in O_N_hr_t_id]

        N_O_hr_t_id = self.data_id(N_O_triple_hr_t)
        N_O_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in N_O_hr_t_id]

        N_N_hr_t_id = self.data_id(N_N_triple_hr_t)
        N_N_tr_h_id = [(i[2], i[1] + 1, i[0]) for i in N_N_hr_t_id]

        # 获取关系分类
        # print(len(set([i[1] for i in O_O_triple_hr_t])))
        # print(len(set([i[1] for i in O_N_triple_hr_t])))
        # print(len(set([i[1] for i in N_O_triple_hr_t])))
        # print(len(set([i[1] for i in N_N_triple_hr_t])))

        return O_O_hr_t_id, O_N_hr_t_id, N_O_hr_t_id, N_N_hr_t_id, O_O_tr_h_id, O_N_tr_h_id, N_O_tr_h_id, N_N_tr_h_id

        # return self.data_id(O_O_triple_hr_t), self.data_id(O_N_triple_hr_t), self.data_id(N_O_triple_hr_t), \
        #        self.data_id(N_N_triple_hr_t), self.data_id(O_O_triple_tr_h), self.data_id(O_N_triple_tr_h), \
        #        self.data_id(N_O_triple_tr_h), self.data_id(N_N_triple_tr_h)


# Data(data_dir="data/FB15k/", reverse=True)
