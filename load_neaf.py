from os import path
import json
import numpy as np
import torch

def load_neaf_data(basedir, ray_file):
    with open(path.join(basedir, ray_file), 'r') as rf:
        loaded_json = json.load(rf)

    listener_count = len(loaded_json['states'])
    tst_cnt = listener_count // 10

    listener_ids = np.arange(0, listener_count)
    np.random.shuffle(listener_ids)
    i_split = [listener_ids[:-2*tst_cnt], listener_ids[-2*tst_cnt:-tst_cnt], listener_ids[-tst_cnt:]]
    bounding_box = (torch.tensor([-3., -3., -3.]), torch.tensor([3., 3., 3.]))
    return loaded_json["states"], i_split, bounding_box


def build_ray_batch(state, args):
    # read stte vals
    d = torch.device('cuda')
    listener_pos = torch.Tensor(state['listener']['matrix'][:3], device=d)
    bounces = [ray["last_bounce"] for ray in state["rays"]]
    bounces = torch.Tensor([bounce for bounce in bounces], device=d)
    directions = bounces - listener_pos if bounces.shape[0] > 0 else []
    directions = directions / torch.linalg.norm(directions, dim=1, keepdims=True)
    distances = torch.Tensor([ray["distance"] for ray in state["rays"]], device=d)
    falloff_factors = 1 / (1 + distances)
    absorption = torch.Tensor([ray["absorption"] for ray in state["rays"]], device=d)
    absorption = torch.stack((torch.mean(absorption[:, :3], dim=1),
                              torch.mean(absorption[:, 3:7], dim=1),
                              torch.mean(absorption[:, 7:], dim=1)), dim=1)
    incoming = 1 - absorption
    incoming = incoming * falloff_factors[:, None]

    # get timing information
    sos = args.speed_of_sound
    delays = distances / sos
    timeinterval = args.time_interval
    timesteps = args.neaf_timesteps
    delays = torch.round(delays * (timesteps * (1 / timeinterval)))

    # build N_rand random receivers
    rec_count = args.N_rand
    rec_times = torch.rand(rec_count)
    rec_times_discrete = torch.round(rec_times * (timesteps * (1 / timeinterval)))
    recs_d = torch.from_numpy(np.random.normal(size=(rec_count, 3)).astype(np.float32))
    recs_d = recs_d / torch.linalg.norm(recs_d, axis=1, keepdims=True)
    recs_d = torch.Tensor(recs_d).to(torch.device('cuda'))
    dots = torch.clamp(torch.einsum('rc,nc->rn', recs_d, directions), 0, 1)
    # TODO modify dots before using
    weighted_incoming = incoming[None, ...] * dots[..., None]

    target = torch.zeros((rec_count, 3))
    for rec in range(rec_count):
        for sp_idx, timestep in enumerate(delays):
            if timestep == rec_times_discrete[rec]:
                target[rec] += weighted_incoming[rec, sp_idx]

    recs_o = listener_pos
    recs_o = torch.broadcast_to(recs_o, recs_d.shape)
    recs = torch.stack((recs_o, recs_d), 0)

    return recs, target, rec_times

