
import logging
import os

class logset:
    @staticmethod
    def set_logger(path,file=None):
        '''
        Write logs to checkpoint and console
        '''
        if file != None:
            log_file = os.path.join(path,file)
        else:
            log_file = os.path.join(path, 'train.log')
        # if args.do_train:
        #     log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
        # else:
        #     log_file = os.path.join(args.save_path or args.init_checkpoint, 'axtest.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        

    @staticmethod
    def log_metrics(mode, step, metrics):
        '''
        Print the evaluation logs
        '''
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
