import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip

import pyscipopt as scip
import utilities


class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, out_queue, exploration_policy, query_expert_prob, out_dir, follow_expert=True):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.out_queue = out_queue
        self.exploration_policy = exploration_policy
        self.query_expert_prob = query_expert_prob
        self.out_dir = out_dir
        self.follow_expert = follow_expert

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinit(self):
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        if self.model.getNNodes() == 1:
            # initialize root buffer for Khalil features extraction
            utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

        # once in a while, also run the expert policy and record the (state, action) pair
        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert:
            state = utilities.extract_state(self.model)
            cands, *_ = self.model.getPseudoBranchCands()
            state_khalil = utilities.extract_khalil_variable_features(self.model, cands, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()

            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])

            action_set = [c.getCol().getLPPos() for c in cands]
            expert_action = action_set[bestcand]

            data = [state, state_khalil, expert_action, action_set, scores]

            # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
            if not any([s < 0 for s in scores]):

                filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': self.episode,
                        'instance': self.instance,
                        'seed': self.seed,
                        'node_number': self.model.getCurrentNode().getNumber(),
                        'node_depth': self.model.getCurrentNode().getDepth(),
                        'data': data,
                        }, f)

                self.out_queue.put({
                    'type': 'sample',
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'filename': filename,
                })

                self.sample_counter += 1

        # if exploration and expert policies are the same, prevent running it twice
        if not query_expert or (not self.follow_expert and self.exploration_policy != 'vanillafullstrong'):
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # apply 'vanillafullstrong' branching decision if needed
        if query_expert and self.follow_expert or self.exploration_policy == 'vanillafullstrong':
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            self.model.branchVar(cands[bestcand])
            result = scip.SCIP_RESULT.BRANCHED

        return {"result": result}


def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    while True:
        episode, instance, seed, exploration_policy, query_expert_prob, time_limit, out_dir = in_queue.get()
        print(f'[w {os.getpid()}] episode {episode}, seed {seed}, processing instance \'{instance}\'...')

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        utilities.init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)

        branchrule = SamplingAgent(
            episode=episode,
            instance=instance,
            seed=seed,
            out_queue=out_queue,
            exploration_policy=exploration_policy,
            query_expert_prob=query_expert_prob,
            out_dir=out_dir)

        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        m.optimize()
        m.freeProb()

        print(f"[w {os.getpid()}] episode {episode} done, {branchrule.sample_counter} samples")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def send_orders(orders_queue, instances, seed, exploration_policy, query_expert_prob, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, exploration_policy, query_expert_prob, time_limit, out_dir])
        episode += 1


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, query_expert_prob, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), exploration_policy, query_expert_prob, time_limit, tmp_samples_dir),
            daemon=True)
    dispatcher.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                    # early stop dispatcher (hard)
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['Standard_MTSP', 'Standard_MTSP_10w', 'MinMax_MTSP', 'MinMax_MTSP_10w', 'Bounded_MTSP', 'Bounded_MTSP_10w'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    train_size = 20000
    valid_size = 4000
    test_size = 4000
    exploration_strategy = 'pscost'
    node_record_prob = 0.05
    time_limit = 3600

    if args.problem == 'Standard_MTSP':
        instances_train = glob.glob('data/instances/Standard_MTSP/train_9_3/*.lp')
        instances_valid = glob.glob('data/instances/Standard_MTSP/valid_9_3/*.lp')
        instances_test = glob.glob('data/instances/Standard_MTSP/test_9_3/*.lp')
        out_dir = 'data/samples/Standard_MTSP/9_3'

    elif args.problem == 'Standard_MTSP_10w':
        train_size = 100000
        valid_size = 20000
        test_size = 20000
        instances_train = glob.glob('data/instances/Standard_MTSP/train_9_3/*.lp')
        instances_valid = glob.glob('data/instances/Standard_MTSP/valid_9_3/*.lp')
        instances_test = glob.glob('data/instances/Standard_MTSP/test_9_3/*.lp')
        out_dir = 'data/samples/Standard_MTSP_10w/9_3'

    elif args.problem == 'MinMax_MTSP':
        instances_train = glob.glob('data/instances/MinMax_MTSP/train_9_3/*.lp')
        instances_valid = glob.glob('data/instances/MinMax_MTSP/valid_9_3/*.lp')
        instances_test = glob.glob('data/instances/MinMax_MTSP/test_9_3/*.lp')
        out_dir = 'data/samples/MinMax_MTSP/9_3'

    elif args.problem == 'MinMax_MTSP_10w':
        train_size = 100000
        valid_size = 20000
        test_size = 20000
        instances_train = glob.glob('data/instances/MinMax_MTSP/train_9_3/*.lp')
        instances_valid = glob.glob('data/instances/MinMax_MTSP/valid_9_3/*.lp')
        instances_test = glob.glob('data/instances/MinMax_MTSP/test_9_3/*.lp')
        out_dir = 'data/samples/MinMax_MTSP_10w/9_3'

    elif args.problem == 'Bounded_MTSP':
        train_size = 100000
        valid_size = 20000
        test_size = 20000
        instances_train = glob.glob('data/instances/Bounded_MTSP/train_12_3/*.lp')
        instances_valid = glob.glob('data/instances/Bounded_MTSP/valid_12_3/*.lp')
        instances_test = glob.glob('data/instances/Bounded_MTSP/test_12_3/*.lp')
        out_dir = 'data/samples/Bounded_MTSP/12_3'

    elif args.problem == 'Bounded_MTSP_10w':
        train_size = 100000
        valid_size = 20000
        test_size = 20000
        instances_train = glob.glob('data/instances/Bounded_MTSP/train_12_3/*.lp')
        instances_valid = glob.glob('data/instances/Bounded_MTSP/valid_12_3/*.lp')
        instances_test = glob.glob('data/instances/Bounded_MTSP/test_12_3/*.lp')
        out_dir = 'data/samples/Bounded_MTSP_10w/12_3'
    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    # create output directory, throws an error if it already exists
    # os.makedirs(out_dir)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, test_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 2)
    collect_samples(instances_test, out_dir + '/test', rng, test_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit)
