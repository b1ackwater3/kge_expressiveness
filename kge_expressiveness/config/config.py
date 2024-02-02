import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_test_class', action='store_true', default=False)  
    parser.add_argument('--test_relation', type=str)  
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_reverse', action='store_true')

    # 模型的配置参数
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('--relation_dim', type=int, default = 0)    
    parser.add_argument('-g', '--gamma', default=12.0, type=float)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
 
    parser.add_argument('--dropout1', type=float, default = 0.2)    
    parser.add_argument('--dropout2', type=float, default = 0.2)    
    parser.add_argument('--dropout3', type=float, default = 0.3)    
    parser.add_argument('--label_smoothing', type=float, default = 0.1)  

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)

    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-loss', '--loss_function', type=str,default='NSSAL')
    parser.add_argument('-optimizer', '--optimizer', type=str,default='Adam')


    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
  # add rotpro
    parser.add_argument('-alpha', '--alpha', default=0.0005, type=float)
    parser.add_argument('-c', '--constrains', default=False)
    parser.add_argument('-gamma_m', '--gamma_m', default=0.000001, type=float)
    parser.add_argument('-min_len', '--min_len', default=0.001, type=float)
    parser.add_argument('-rel_gam', '--rel_gam', default=0.001, type=float)

    parser.add_argument('-ralpha', '--ralpha', default=0.00, type=float)
    parser.add_argument('-rgamma_m', '--rgamma_m', default=0.001, type=float)
    parser.add_argument('-rbeta', '--rbeta', default=0.001, type=float)
    
    parser.add_argument('-beta', '--beta', default=1.5, type=float)
    
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    # add rotpro
    parser.add_argument('--init_pr', default=1.0, type=float,
                        help='constrain relational rotation phase when initialization')
    parser.add_argument('--train_pr',  default=1.0, type=float,
                        help='constrain relational rotation phase when training')
    parser.add_argument('--trans_test', type=str, default='test.txt', help='test file on transitive triples.') 
    parser.add_argument('--do_rel_constrain', action='store_true')

    # para for training
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--lr_step', default=None, type=int)
    parser.add_argument("--decay", type=float, default=0.1, nargs="?", help="Decay rate.")

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--transh_dropout', type=float, default=1)

    parser.add_argument('--constype', type=str, default="exp6")



    # 增加预先训练的方法:输入是预训练的模型
    parser.add_argument('--pre_train', type=str, default=None) 
    # Negative sampling or 1-n training   
    parser.add_argument('--train_type', type=str, default="NagativeSample")    
    return parser.parse_args(args)

def parse_simple(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_relation', type=str)  


    # 模型的配置参数
    parser.add_argument('--model', default='TransE', type=str)
  
    
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')


    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
   
 
    parser.add_argument('--save_path', type=str)    
    return parser.parse_args(args)
# def parse_args_test(args=None):
#     parser = argparse.ArgumentParser(
#         description='Training and Testing Knowledge Graph Embedding Models',
#         usage='train.py [<args>] [-h | --help]'
#     )

#     parser.add_argument('--cuda', action='store_true', help='use GPU')
    
#     parser.add_argument('--do_train', action='store_true',default=True)
#     parser.add_argument('--do_valid', action='store_true',default=True)
#     parser.add_argument('--do_test', action='store_true',default=True)
#     parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
#     parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
#     parser.add_argument('--regions', type=int, nargs='+', default=None, 
#                         help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
#     parser.add_argument('--data_path', type=str, default="D:/2021_project/KG_learn/KGE_tool/data/YAGO3-10")
#     parser.add_argument('--model', default='TransE', type=str)
#     parser.add_argument('-de', '--double_entity_embedding', action='store_true')
#     parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
#     parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
#     parser.add_argument('-d', '--hidden_dim', default=500, type=int)
#     parser.add_argument('-g', '--gamma', default=12.0, type=float)
#   # add rotpro
#     parser.add_argument('-alpha', '--alpha', default=0.25, type=float)

#     parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
#     parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
#     parser.add_argument('-b', '--batch_size', default=1024, type=int)
#     parser.add_argument('-r', '--regularization', default=0.0, type=float)
#   # add rotpro
#     parser.add_argument('-c', '--constrains', default=False)
#     parser.add_argument('-gamma_m', '--gamma_m', default=0.00005, type=float)
#     parser.add_argument('-beta', '--beta', default=1.5, type=float)
    
#     parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
#     parser.add_argument('--uni_weight', action='store_true', 
#                         help='Otherwise use subsampling weighting like in word2vec')
    
#     # add rotpro
#     parser.add_argument('--init_pr', default=1.0, type=float,
#                         help='constrain relational rotation phase when initialization')
#     parser.add_argument('--train_pr',  default=1.0, type=float,
#                         help='constrain relational rotation phase when training')
#     parser.add_argument('--trans_test', type=str, default='test.txt', help='test file on transitive triples.')

#     # para for training
#     parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
#     parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
#     parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
#     parser.add_argument('-save', '--save_path', default='/models/TransE_00/', type=str)
#     parser.add_argument('--max_steps', default=100000, type=int)
#     parser.add_argument('--warm_up_steps', default=None, type=int)
#     parser.add_argument("--decay", type=float, default=0.1, nargs="?", help="Decay rate.")

    
#     parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
#     parser.add_argument('--valid_steps', default=10000, type=int)
#     parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
#     parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    
#     parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
#     parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

#     parser.add_argument('--test_relation', type=str)     
#     # 对于多个关系的参数的处理
#     parser.add_argument('--rel_dim', type=int, default=0)    
#     # 增加预先训练的方法:输入是预训练的模型
#     parser.add_argument('--pre_train', type=str, default=None) 
#     # Negative sampling or 1-n training   
#     parser.add_argument('--train_mode', type=str, default="NagativeSample")    
#     return parser.parse_args(args)

