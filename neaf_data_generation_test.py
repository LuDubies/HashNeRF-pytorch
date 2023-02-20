import torch
import pytest
from neaf_operations import build_neaf_batch, load_neaf_data
from argparse import Namespace

@pytest.fixture(autouse=True)
def run_around_tests():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    yield

def test_load_neaf_data():
    states, i_split, bb = load_neaf_data(r'./test_data', r'rays_clean_200.json')
    assert(len(states) == 200)
    assert(len(i_split) == 3)

def test_batch_building():
    args = {'neaf_timesteps': 100,
            'time_interval': 0.05,
            'speed_of_sound': 343.,
            'N_rand': 2048,
            'angle_exp': 2}
    args = Namespace(**args)

    listener_states, _, _ = load_neaf_data(r'./test_data', r'rays_clean_200.json')

    test_loc_i = 0
    algo_test_recs, algo_test_ir = build_neaf_batch(listener_states, [test_loc_i], args, reccount=100, mode='ir')
    algo_test_recs_2, algo_test_target, algo_test_times = build_neaf_batch(listener_states, [test_loc_i], args,
                                                                           reccount=100, mode='rec',
                                                                           directions=algo_test_recs[1])
    assert(torch.allclose(algo_test_recs, algo_test_recs_2))

    error = False
    algo_test_times = torch.floor(algo_test_times * args.neaf_timesteps)
    for test_iter in range(algo_test_times.shape[0]):
        time_indx = int(algo_test_times[test_iter])
        if not torch.allclose(algo_test_target[test_iter], algo_test_ir[test_iter, time_indx]):
            print(f"Error for it {test_iter}.")
            error = True
    assert(not error)


