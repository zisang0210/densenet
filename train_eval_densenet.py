from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset_name', type=str, default='quiz')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model_name', type=str, default='densenet')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay_type', type=str, default='exponential')
    parser.add_argument('--num_epochs_per_decay', type=int, default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.5)
    parser.add_argument('--clone_on_cpu', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)

    # eval
    parser.add_argument('--dataset_split_name', type=str, default='validation')
    parser.add_argument('--eval_dir', type=str, default='validation')
    parser.add_argument('--max_num_batches', type=int, default=None)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


train_cmd = 'python ./train_image_classifier.py  --dataset_name={dataset_name} --dataset_dir={dataset_dir} --model_name={model_name} --train_dir={train_dir} --learning_rate={learning_rate} --learning_rate_decay_type={learning_rate_decay_type} --num_epochs_per_decay={num_epochs_per_decay} --learning_rate_decay_factor={learning_rate_decay_factor} --weight_decay={weight_decay} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    step_per_epoch = 50000 // FLAGS.batch_size
    for i in range(10):
        steps = int(step_per_epoch * (i + 1))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                         'model_name': FLAGS. model_name, 'train_dir': FLAGS. train_dir,
                                         'learning_rate': FLAGS.learning_rate, 'learning_rate_decay_type': FLAGS.learning_rate_decay_type,
                                         'num_epochs_per_decay': FLAGS.num_epochs_per_decay, 'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
                                         'weight_decay': FLAGS.weight_decay,'optimizer': FLAGS.optimizer,
                                         'batch_size': FLAGS.batch_size, 'max_number_of_steps': steps, 'clone_on_cpu':FLAGS.clone_on_cpu}))
        for l in p:
            print(p.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                        'dataset_split_name': FLAGS.dataset_split_name, 'model_name': FLAGS. model_name,
                                        'checkpoint_path': FLAGS.train_dir, 'batch_size': FLAGS.batch_size,
                                        'eval_dir': FLAGS. eval_dir, 'max_num_batches': FLAGS. max_num_batches}))
        for l in p:
            print(p.strip())
