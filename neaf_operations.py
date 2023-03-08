from os import path
import json
import numpy as np
import torch


from ir_visualization import cgrade_ir, error_plot

def load_neaf_data(basedir, ray_file):
    with open(path.join(basedir, ray_file), 'r') as rf:
        loaded_json = json.load(rf)

    listener_count = len(loaded_json['states'])
    tst_cnt = listener_count // 10

    listener_ids = np.arange(0, listener_count)
    np.random.shuffle(listener_ids)
    i_split = [listener_ids[:-tst_cnt], listener_ids[-tst_cnt:], []]

    if 'source' in loaded_json.keys():
        source_pos = torch.Tensor(loaded_json['source']['matrix'][:3])
    else:
        source_pos = None
    bounding_box = (torch.tensor([-3., -3., -3.]), torch.tensor([3., 3., 3.]))
    return loaded_json["states"], i_split, source_pos, bounding_box

def build_neaf_batch(states, listener_ids, args, reccount=None, mode='rec', directions=None):
    # random directions for receivers
    if reccount is None:
        reccount = args.N_rand
    if directions is None:
        recs_d = get_random_directions(reccount).to(torch.device('cuda'))
    else:
        recs_d = directions
    if mode == 'rec':
        return build_rec_batch(states, listener_ids, reccount, recs_d, args)
    elif mode == 'ir':
        return build_ir_batch(states, listener_ids, reccount, recs_d, args)
    else:
        raise NotImplementedError()

def build_rec_batch(states, listener_ids, reccount, recs_d, args):
    if len(listener_ids) == 1:
        # no need to build a batch from different locations
        return get_random_receivers_for_listener(states[listener_ids[0]], reccount, recs_d, args)
    recs_per_loc = int(np.ceil(reccount / len(listener_ids)))
    recs = []
    targets = []
    times = []
    rec_i = 0
    for lid in listener_ids:
        if rec_i + recs_per_loc <= recs_d.shape[0]:
            re, ta, ti = get_random_receivers_for_listener(states[lid], recs_per_loc,
                                                           recs_d[rec_i:rec_i+recs_per_loc], args)
        else:
            re, ta, ti = get_random_receivers_for_listener(states[lid], recs_d[rec_i:].shape[0],
                                                           recs_d[rec_i:], args)
        rec_i += recs_per_loc
        recs.append(re)
        targets.append(ta)
        times.append(ti)
    recs = torch.cat(recs, dim=1)
    targets = torch.cat(targets, dim=0)
    times = torch.cat(times, dim=0)

    return recs[:, :reccount, :], targets[:reccount, ...], times[:reccount]


def get_random_receivers_for_listener(state, rec_count, recs_d, args):
    # build empy target
    target = torch.zeros((rec_count, 3))
    # random times
    rec_times = torch.rand(rec_count)

    # read state vals
    listener_pos, directions, distances, incoming = load_data_for_state(state)

    # build hashNerf ray format for receivers
    recs_o = listener_pos
    recs_o = torch.broadcast_to(recs_o, recs_d.shape)
    recs = torch.stack((recs_o, recs_d), 0)  # shape (2, recs, 3)

    if distances is None:
        return recs, target, rec_times
    delays = calculate_delays(distances, args)

    # build N_rand random receivers
    rec_times_discrete = torch.floor(rec_times * args.neaf_timesteps)
    weighted_incoming = calculate_incoming_impulse(recs_d, directions, incoming, args)  # shape (recs, sp_rays, 3)

    # get relevant rays for each recorder time
    relevancy = torch.eq(rec_times_discrete[:, None], delays[None, :]).float()  # shape (recs, sp_rays)
    weighted_incoming = weighted_incoming * relevancy[:, :, None]
    target = torch.sum(weighted_incoming, dim=1)  # shape (recs, 3)

    return recs, target, rec_times


def build_ir_batch(states, listener_ids, reccount, recs_d, args):
    if len(listener_ids) == 1:
        # no need to build a batch from different locations
        return get_random_ir_for_listener(states[listener_ids[0]], reccount, recs_d, args)
    recs_per_loc = int(np.ceil(reccount / len(listener_ids)))
    recs = []
    irs = []
    rec_i = 0
    for lid in listener_ids:
        if rec_i + recs_per_loc < recs_d.shape[0]:
            re, ir = get_random_ir_for_listener(states[lid], recs_per_loc, recs_d[rec_i: rec_i+recs_per_loc], args)
        else:
            re, ir = get_random_ir_for_listener(states[lid], recs_d[rec_i:].shape[0], recs_d[rec_i:], args)
        rec_i += recs_per_loc
        recs.append(re)
        irs.append(ir)
    recs = torch.cat(recs, dim=1)
    irs = torch.cat(irs, dim=0)
    irs = irs.cpu().numpy()
    return recs[:, :reccount, :], irs[:reccount, ...]

def get_random_ir_for_listener(state, rec_count, recs_d, args):
    # build empy target
    irs = torch.zeros((rec_count, args.neaf_timesteps, 3))
    # random times
    rec_intervals = torch.arange(args.neaf_timesteps)

    # read state vals
    listener_pos, directions, distances, incoming = load_data_for_state(state)

    # build hashNerf ray format for receivers
    recs_o = listener_pos
    recs_o = torch.broadcast_to(recs_o, recs_d.shape)
    recs = torch.stack((recs_o, recs_d), 0)  # shape (2, recs, 3)

    if distances is None:
        return recs, irs
    delays = calculate_delays(distances, args)  # shape (sp_rays)
    weighted_incoming = calculate_incoming_impulse(recs_d, directions, incoming, args)  # shape (recs, sp_rays, 3)

    interval_matching = torch.eq(rec_intervals[:, None], delays[None, :]).float()  # shape(timesteps, sp_rays)
    irs = torch.einsum('rsn,ts->rtn', weighted_incoming, interval_matching)

    return recs, irs


def get_random_directions(count):
    directions = torch.from_numpy(np.random.normal(size=(count, 3)).astype(np.float32))
    directions = directions / torch.linalg.norm(directions, axis=1, keepdims=True)
    return torch.Tensor(directions)

def load_data_for_state(state):
    listener_pos = torch.Tensor(state['listener']['matrix'][:3])
    bounces = [ray["last_bounce"] for ray in state["rays"]]
    if bounces:
        bounces = torch.Tensor([bounce for bounce in bounces])
        directions = bounces - listener_pos

        directions = directions / torch.linalg.norm(directions, dim=1, keepdims=True)
        distances = torch.Tensor([ray["distance"] for ray in state["rays"]])
        falloff_factors = 1 / (1 + distances)
        absorption = torch.Tensor([ray["absorption"] for ray in state["rays"]])
        absorption = torch.stack((torch.mean(absorption[:, :3], dim=1),
                                  torch.mean(absorption[:, 3:7], dim=1),
                                  torch.mean(absorption[:, 7:], dim=1)), dim=1)
        incoming = 1 - absorption
        incoming = incoming * falloff_factors[:, None]
        return listener_pos, directions, distances, incoming
    return listener_pos, None, None, None

def calculate_delays(distances, args):
    sos = args.speed_of_sound
    delays = distances / sos
    # split timeinterval into timesteps "containers"/indices
    timeinterval = args.time_interval
    timesteps = args.neaf_timesteps
    return torch.floor(delays * (timesteps * (1 / timeinterval)))


def calculate_incoming_impulse(receiver_directions, ray_directions, incoming, args):
    dots = torch.clamp(torch.einsum('rc,nc->rn', receiver_directions, ray_directions), 0, 1)  # shape (recs, sp_rays)
    dots = torch.pow(dots, args.angle_exp)
    return incoming[None, ...] * dots[..., None]  # shape (recs, sp_rays, 3)






