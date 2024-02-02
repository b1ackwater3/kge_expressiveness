import os 
from collections import defaultdict
import numpy as np

class Relation2Patter:

    trans = ['isLocatedIn', 'isConnectedTo', 'locatedin', '/location/location/contains', 'location_of', 'contains',
        'connected_to', 'associated_with',  
        'concept:istallerthan', 'concept:locationlocatedwithinlocation']
    symmetric = ['isMarriedTo', 'neighbor', '/people/person/spouse_s./people/marriage/spouse',
    '/location/location/adjoin_s./location/adjoining_relationship/adjoins',
    '/people/person/sibling_s./people/sibling_relationship/sibling',
    'interacts_with', 'connected_to', 'associated_with', 'interconnects', 'adjacent_to',
    'concept:hassibling', 'concept:hasspouse']
    subProperty = {'wasBornOnDate': 'startsExistingOnDate',
            'wasDestroyedOnDate': 'endsExistingOnDate'}
    functional = ['hasISIN', 'hasGini', 'hasAirportCode', 'occursSince', 'endsExistingOnDate', 'hasDuration',
    'happenedOnDate', 'hasLength', 'hasISBN', 'diedIn', 'hasGender', 'hasCapital', 'wasBornIn',
    'hasArea',
    '/people/deceased_person/place_of_death', '/location/country/capital',
    '/people/deceased_person/place_of_burial', '/people/person/gender',
    '/organization/organization/place_founded','/location/hud_county_place/county', 
    'isa',
    'concept:persondiedincountry', 'concept:personborninlocation', 'concept:organizationheadquarteredincity']
    asymmetric = ['/film/director/film', '/film/film/costume_design_by',
    '/film/film/dubbing_performances./film/dubbing_performance/actor',
    '/location/capital_of_administrative_division/capital_of./location/administrative_division_capital_relationship/administrative_division',
    '/people/person/places_lived./people/place_lived/location', '/film/film/story_by',
    'concept:hashusband', 'concept:haswife', 'concept:fatherofperson', 'concept:motherofperson',
    'concept:parentofperson', 'concept:bankboughtbank', 'concept:museumincity',
    'concept:personleadsorganization',   
    'concept:worksfor', 'concept:agentbelongstoorganization','concept:athletehomestadium',   
    'concept:athleteplaysforteam', 'concept:athleteplaysinleague', 'concept:teamplaysinleague',   
    'concept:teamplayssport','concept:personborninlocation','concept:athleteplayssport',   
    'concept:organizationhiredperson','concept:organizationheadquarteredincity',
    'isa','wasBornIn', 'worksAt', 'happenedIn', 'isLeaderOf', 'hasWonPrize', 'isAffiliatedTo', 'hasChild',
     'exports', 'isLocatedIn', 'hasCurrency', 'created', 'graduatedFrom', 'actedIn', 'hasGender',
    'hasCapital', 'isInterestedIn', 'influences', 'owns', 'isKnownFor', 'directed', 'diedIn', 'livesIn', 'isPoliticianOf',
    'hasWebsite', 'hasOfficialLanguage', 'hasAcademicAdvisor', 'isCitizenOf', 'edited', 'playsFor', 'participatedIn',
    'wroteMusicFor', 'hasMusicalRole','_has_part','_member_meronym','_member_of_domain_region','_member_of_domain_usage']

class DataProcesser:

    def __init__(self,data_path, dataType="Normal" ,idDict=True, is_id=False, reverse=False):
        super(DataProcesser, self).__init__()

        self.data_path = data_path
        self.dataType = dataType
        self.entity2id,self.relation2id = None, None

        if idDict:
            self.entity2id,self.relation2id = DataProcesser.read_idDic(self.data_path)
            self.nentity = len(self.entity2id)
            self.nrelation = len(self.relation2id)
            self.idDict = True
            self.id2relation = {
                self.relation2id[k]:k for k in self.relation2id.keys()
            }
            self.id2entity = {
                self.entity2id[k]:k for k in self.entity2id.keys()
            }

        else:
            self.idDict = False

        if dataType == 'Normal':
            self.read_normal_data(is_id)
        elif dataType == 'ClassTest':
            self.read_classtest_data(is_id)

        if is_id or self.idDict :
            self.triples_type = 'Id'
        else:
            self.triples_type = 'Raw'

        if reverse:
            # 样本增加reverse
            self.add_reverse()
            if self.idDict:
                extend_dic ={}
                for r in self.relation2id.keys():
                    extend_dic[r+"_reverse"] = self.relation2id[r] + self.nrelation
                self.relation2id.update(extend_dic)
                self.nrelation *= 2

        if self.idDict == False and is_id == False:
            self.build_dict()
            self.trans_to_id()
    
    def build_dict(self):
        self.entities = sorted(list(set([h for h,r,t in self.all_true_triples ] + [t for h,r,t in self.all_true_triples])))
        self.relations = sorted(list(set([r for h,r,t in self.all_true_triples])))
        self.nentity = len(self.entities)
        self.nrelation = len(self.relations)

        self.entity2id = {
            self.entities[id]:id for id in range(self.nentity)
        }
        self.relation2id = {
           
        }
        self.id2entity = {
            id:self.entities[id] for id in range(self.nentity)
        }
        self.id2relation={
             id:self.relations[id] for id in range(self.nrelation)
        }

        self.idDict = True

    def trans_to_id(self):
        trans_train = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) for h,r,t in self.train]
        trans_valid = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) for h,r,t in self.train]
        trans_test = [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) for h,r,t in self.train]

        self.train, self.valid, self.test = trans_train,trans_valid,trans_test
        self.all_true_triples = self.train + self.valid + self.test
        self.triples_type = 'Id'

    
    def get_reverse_train(self):
        reverse_train = []
        for h,r,t in self.train:
            reverse_train.append((t,r,h))
        return reverse_train


    def build_transfor_test_filter(self):
        all_true =set(self.all_true_triples)
        h2ts = defaultdict(set)
        for h,r,t in all_true:
            h2ts[h].add(t)
        
        trans_close = set()
        
        for h,r,t in all_true:
            trans_close.add((h,r,t))
            for second_t in h2ts[t]:
                if (h,r,second_t) not in all_true:
                    trans_close.add((h,r,second_t))

        self.all_true_triples = trans_close


    def add_reverse2list(self, raw):
        gen_triples = []
        for h,r,t in raw:
            if self.triples_type == 'Id':
                gen_triples.append((t, r+self.nrelation, h))
            else:
                gen_triples.append((t, r+'_reverse', h))
        return gen_triples

    def add_reverse(self):
        if self.dataType == 'Normal':
            self.train.extend(self.add_reverse2list(self.train))
            self.test.extend(self.add_reverse2list(self.test))
            self.all_true_triples = self.train + self.test 
            self.valid.extend(self.add_reverse2list(self.valid))
            self.all_true_triples = self.train + self.valid + self.test 
        else:
            pass
        
    def get_classtest_list(self):
        if self.dataType != "ClassTest":
            raise ValueError('Only ClassTest Dataset can get ClassTest List')
        [self.class_test_data[relation_type] for relation_type in ['symmetry','asymmetry','inverse','transitive','composition']]

    def read_normal_data(self,is_id):
        self.train = DataProcesser.read_triples(os.path.join(self.data_path,'train.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.test =  DataProcesser.read_triples(os.path.join(self.data_path,'test.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.valid = DataProcesser.read_triples(os.path.join(self.data_path,'valid.txt'),self.entity2id,self.relation2id,is_id=is_id)
        self.all_true_triples = self.train + self.valid + self.test
    
    def read_classtest_data(self,is_id):
        pre = 'extend'
        relation_types = ['symmetry','asymmetry','inverse','transitive','composition']
        # relation_types = ['symmetry']
        types_files = {
            'symmetry':['pre.txt','gen.txt'],
            'asymmetry':['pre.txt','gen.txt'],
            'inverse':['pre.txt','gen.txt'],
            'transitive':['pre1.txt','pre2.txt','gen.txt'],
            'composition':['pre1.txt','pre2.txt','gen.txt'],
        }
        self.class_test_data = {}
        for relation_type in relation_types:
            triples = []
            for file_back in types_files[relation_type]:
                file_path = os.path.join(self.data_path, pre,relation_type+"_"+file_back)
                tripls = DataProcesser.read_triples(file_path,self.entity2id,self.relation2id,is_id=is_id,distinct=False)
                triples.append(tripls)
            self.class_test_data[relation_type] = triples
        file_path = os.path.join(self.data_path,"transitive_all.txt" )
        self.all_transtive =  DataProcesser.read_triples(file_path,self.entity2id,self.relation2id,is_id=is_id,distinct=False)

        # 统计其中的对称关系和传递关系
        self.symmetry_relation = set()
        for h,r,t in self.class_test_data["symmetry"][0]:
            self.symmetry_relation.add(r)

        self.transitive_relation = set()
        for h,r,t in self.class_test_data["transitive"][0]:
            self.transitive_relation.add(r)
        

    @staticmethod
    def read_idDic(data_path):
        with open(os.path.join(data_path, 'entities.dict'),encoding='utf-8') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
        with open(os.path.join(data_path, 'relations.dict'),encoding='utf-8') as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)
        return entity2id,relation2id

    def find_functional(triples):
        relation = set()
        rel2triples = defaultdict(list)
        for h,r,t in triples:
            relation.add(r)
            rel2triples[r].append((h,r,t))

        for r in relation:
            reltriples = rel2triples[r]
            h2t = defaultdict(list)
            isOk = True
            for h,r,t in reltriples:
                h2t[h].append(t)
                if len(h2t[h]) > 1:
                    isOk = False
                    break
            if isOk:
                print(r)



    @staticmethod
    def read_triples(file_path, entity2id=None, relation2id=None, is_id=False, distinct = True):
        if distinct:
            triples = set()
        else:
            triples = []
        is_tran = False
        if entity2id is not None and relation2id is not None:
            is_tran = True
        elif entity2id is None and relation2id is None:
            is_tran = False
        else:
            raise ValueError("entity2id and relation2id should both be None or not be None")
        with open(file_path, encoding='utf-8') as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')  

                if distinct:
                    if is_id:
                        triples.add((int(h),int(r),int(t)))
                    elif is_tran:
                        if h not in entity2id or t not in entity2id or r not in relation2id:
                            continue
                        triples.add((entity2id[h], relation2id[r], entity2id[t]))
                    else:
                        triples.add((h,r,t))
                else:
                    if is_id:
                        triples.append((int(h),int(r),int(t)))
                    elif is_tran:
                        triples.append((entity2id[h], relation2id[r], entity2id[t]))
                    else:
                        triples.append((h,r,t))
        return list(triples)

    @staticmethod
    def build_trans_triple(triples, hr2triples, rt2triples, filter):
        new_build_triples = []
        triples = set(triples)
        print("Trans Caculate begin")
        new_triples = set()
        for h,r,t in triples:
            new_triples.add((h,r,t))
            for triple2 in hr2triples[(t,r)]:
                if (h,r,triple2[2]) not in triples and (h,r,triple2[2]) not in new_triples and \
                (h,r,triple2[2]) not in filter:
                    new_triples.add((h,r,triple2[2]))
                    new_build_triples.append((h,r,triple2[2]))

            for triple2 in rt2triples[(r,h)]:
                if (triple2[0],r,t) not in triples and (triple2[0],r,t) not in new_triples \
                and (triple2[0],r,t) not in filter:
                    new_triples.add((triple2[0],r,t))
                    new_build_triples.append((triple2[0],r,t))
        triples = new_triples
        print("Trans Caculate over")
        return new_build_triples

    def getConSetData(self):
        return self.pos_conf_triples,self.pos_sys_triples,self.pos_trans_triples

    def getVioData(self):
        return self.vio_triples, self.asy_triples,self.functional_triples
    
    def get_self_test(self):
        return self.test_pos_conf_triples, self.test_vio_triples

    def data_aug_test_relation_patter(self):
        if len(self.trained) < 100:
            print("Check Code")
            return 
        self.test_pos_conf_triples=[]
        self.test_pos_trans_triples=[]
        self.test_trans_triples = []

        self.test_asy_triples=[]
        self.test_functional_triples=[]
        self.test_vio_triples = []

        h2triples = defaultdict(set)
        t2triples = defaultdict(set)

        for h,r,t in (list(self.trained) + self.test + self.valid):
            if self.id2relation[r] in self.rels_transitive:
                if (h,r,t) not in self.trained:
                    h2triples[(h,r)].add((h,r,t))
                    t2triples[(r,t)].add((h,r,t))

        for h,r,t in (self.test + self.valid):
            if self.id2relation[r] in self.rels_symmetric:
                if (h,r,t) not in self.trained:
                    self.test_pos_conf_triples.append((h,r,t))
                if (t,r,h) not in self.trained:
                    self.test_pos_conf_triples.append((t,r,h))

            if self.id2relation[r] in self.rels_transitive:
                if (h,r,t) not in self.trained:
                    self.test_trans_triples.append((h,r,t))
                    h2triples[(h,r)].add((h,r,t))
                    t2triples[(r,t)].add((h,r,t))

            if self.id2relation[r] in self.rels_asymmetric:
                self.test_asy_triples.append((h,r,t))
                self.test_vio_triples.append((t,r,h))

            if self.id2relation[r] in self.rels_functional:
                self.test_functional_triples.append((h, r, t))
                # self.test_vio_triples.append((h,r,t))
        for h,r,t in self.test_functional_triples:
            rand_t = np.random.randint(0, self.nentity)
            while(rand_t == t):
                rand_t = np.random.randint(0, self.nentity)
            self.test_vio_triples.append((h,r,rand_t))

        self.test_pos_trans_triples = self.test_trans_triples
        self.test_pos_trans_triples.extend(DataProcesser.build_trans_triple(self.test_trans_triples + self.trans_triples,h2triples,t2triples,set(self.trained)))
        self.test_pos_trans_triples = self.test_pos_trans_triples[0:500]
        self.test_pos_conf_triples.extend(self.test_pos_trans_triples)
        print("Test conf Triples ",len(self.test_pos_conf_triples))
        
    def data_aug_relation_pattern(self):

        self.rels_transitive = set(Relation2Patter.trans)
        self.rels_symmetric = set(Relation2Patter.symmetric)
        self.rels_functional = set(Relation2Patter.functional)
        self.rels_sub_property = dict(Relation2Patter.subProperty)
        self.rels_asymmetric = set(Relation2Patter.asymmetric)

        self.rels_conformance = list(set(Relation2Patter.trans).union(set(Relation2Patter.symmetric)).union(Relation2Patter.subProperty))
        self.rels_violation = list(set(Relation2Patter.functional).union(set(Relation2Patter.asymmetric)))

        self.pos_conf_triples=[]
        self.pos_sys_triples=[]
        self.pos_trans_triples=[]

        self.trans_triples = []

        self.asy_triples=[]
        self.functional_triples=[]
        self.vio_triples =[]

        hr2triples = defaultdict(set)
        rt2triples = defaultdict(set)

        fun_rel_ids = set()

        # 针对训练集进行增强
        for h,r,t in self.train:
            # 如果是对称关系且其生成的没有在测试集中出现过，则ok
            if self.id2relation[r] in self.rels_symmetric:
                if (t,r,h) not in self.test and (t,r,h) not in self.valid:
                    self.pos_conf_triples.append((t,r,h))
                    self.pos_sys_triples.append((t,r,h))
            # 如果是传递关系则直接加入
            if self.id2relation[r] in self.rels_transitive:
                self.trans_triples.append((h,r,t))
                hr2triples[(h,r)].add((h,r,t))
                rt2triples[(r,t)].add((h,r,t))
            # 如果是反对称关系，则直接加入非一致性，不增强样本
            if self.id2relation[r] in self.rels_asymmetric:
                self.asy_triples.append((h,r,t))
            # 如果是函数关系，则直接加入非一致性，也不增强样本
            if self.id2relation[r] in self.rels_functional:
                fun_rel_ids.add(r)
                self.functional_triples.append((h, r, t))

            if self.id2relation[r] in self.rels_violation:
                self.vio_triples.append((h, r, t))
        
        # 传递关系的生成：要过滤掉测试集合中的元组
        self.pos_trans_triples = DataProcesser.build_trans_triple(self.trans_triples,hr2triples,rt2triples,set(self.test+self.valid))
        self.pos_conf_triples.extend(self.pos_trans_triples)

        print("-------------------------data size------------------------")
        print("Sysmetry: %d" % len(self.pos_sys_triples))
        print("Transitive: %d" % len(self.pos_trans_triples))
        print("Asymetry: %d" % len(self.asy_triples))
        print("Functional: %d" % len(self.functional_triples))

        self.fun_rel_idset =fun_rel_ids
        self.trained = set(self.train + self.pos_conf_triples)
        self.train = list(self.trained)
        self.test_filter = set(list(self.trained) + self.valid + self.test)


class DataTool:

    @staticmethod
    def splitTestSetByRelation(path,fileName):
        rel2triples = defaultdict(list)
        with open(os.path.join(path,fileName),'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                h,r,t = line.strip().split('\t')
                rel2triples[r].append((h,r,t))
        fin.close()

        save_path = os.path.join(path, 'split')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(os.path.join(save_path,'relations.txt'),'w', encoding='utf-8') as fout:
            for rel in rel2triples.keys():
                fout.write("%s\n" % rel)
        fout.close()
        for rel in rel2triples.keys():
            with open(os.path.join(save_path, rel),'w',encoding='utf-8') as fout:
                for triple in rel2triples[rel]:
                    fout.write("%s\t%s\t%s\n" % (triple[0],triple[1],triple[2]))
            fout.close() 


    @staticmethod
    def readSplitRelations(path):
        relations = []
        with open(os.path.join(path,'relations.txt'),'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                relations.append(line.strip())
        rel2triples = defaultdict(list)
        for rel in relations:
            with open(os.path.join(path,rel),'r', encoding='utf-8') as fin:
                lines = fin.readlines()
                for line in lines:
                    h,r,t = line.strip().split('\t')
                    rel2triples[r].append((h,r,t))

        return relations, rel2triples
