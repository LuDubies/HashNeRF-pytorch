from os import path
import json
import numpy as np
import torch
import PIL.Image as im

def load_neaf_data(basedir, ray_file):
    with open(path.join(basedir, ray_file), 'r') as rf:
        loaded_json = json.load(rf)

    listener_count = len(loaded_json['states'])
    tst_cnt = listener_count // 10

    listener_ids = np.arange(0, listener_count)
    np.random.shuffle(listener_ids)
    i_split = [listener_ids[:-tst_cnt], listener_ids[-tst_cnt:], []]
    bounding_box = (torch.tensor([-3., -3., -3.]), torch.tensor([3., 3., 3.]))
    return loaded_json["states"], i_split, bounding_box


def build_rec_batch(states, listener_ids, args, reccount=None, full_gt=False):
    if reccount is None:
        reccount = args.N_rand
    if len(listener_ids) == 1:
        # no need to build a batch from different locations
        return get_random_receivers_for_listener(states[listener_ids[0]], reccount, args)
    recs_per_loc = int(np.ceil(reccount / len(listener_ids)))
    recs = []
    targets = []
    times = []
    for lid in listener_ids:
        re, ta, ti = get_random_receivers_for_listener(states[lid], recs_per_loc, args)
        recs.append(re)
        targets.append(ta)
        times.append(ti)
    recs = torch.cat(recs, dim=1)
    targets = torch.cat(targets, dim=0)
    times = torch.cat(times, dim=0)

    return recs[:, :reccount, :], targets[:reccount, ...], times[:reccount]


def get_random_receivers_for_listener(state, rec_count, args, full_gt=False):
    d = torch.device('cuda')

    target = torch.zeros((rec_count, 3))
    rec_times = torch.rand(rec_count)
    recs_d = torch.from_numpy(np.random.normal(size=(rec_count, 3)).astype(np.float32))
    recs_d = recs_d / torch.linalg.norm(recs_d, axis=1, keepdims=True)
    recs_d = torch.Tensor(recs_d).to(torch.device('cuda'))

    # read state vals
    listener_pos = torch.Tensor(state['listener']['matrix'][:3], device=d)
    bounces = [ray["last_bounce"] for ray in state["rays"]]
    if bounces:
        bounces = torch.Tensor([bounce for bounce in bounces], device=d)
        directions = bounces - listener_pos

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
        # split timeinterval into timesteps "containers"/indices
        timeinterval = args.time_interval
        timesteps = args.neaf_timesteps
        delays = torch.round(delays * (timesteps * (1 / timeinterval)))

        # build N_rand random receivers
        rec_times_discrete = torch.round(rec_times * timesteps)
        dots = torch.clamp(torch.einsum('rc,nc->rn', recs_d, directions), 0, 1)  # shape (recs, sp_rays)
        # TODO modify dots before using
        weighted_incoming = incoming[None, ...] * dots[..., None]  # shape (recs, sp_rays, 3)

        # get relevant rays for each recorder time
        relevancy = torch.eq(rec_times_discrete[:, None], delays[None, :]).float()  # shape (recs, sp_rays)
        weighted_incoming = weighted_incoming * relevancy[:, :, None]
        target = torch.sum(weighted_incoming, dim=1)  # shape (recs, 3)

    recs_o = listener_pos
    recs_o = torch.broadcast_to(recs_o, recs_d.shape)
    recs = torch.stack((recs_o, recs_d), 0)  # shape (2, recs, 3)

    return recs, target, rec_times


def save_ir(irs, recs, iteration, savedir, truth=None):
    pil_image = im.fromarray(np.uint8(irs * 255))
    pil_image.save(path.join(savedir, f"ir_{iteration}.png"))
    if truth is not None:
        pil_image = im.fromarray(np.uint8(truth * 255))
        pil_image.save(path.join(savedir, f"ir_truth.png"))
