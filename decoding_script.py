import argparse
import numpy as np
import pickle
import os
from datetime import datetime

import sklearn.svm as skm
import sklearn.neighbors as sknn

import general.data_io as gio
import navigation_position.auxiliary as npa
import navigation_position.representational_analysis as npra
import navigation_position.visualization as npv


def create_parser():
    parser = argparse.ArgumentParser(description='decoding analysis on navigation data')
    parser.add_argument('-o', '--output_folder', default='.', type=str,
                        help='folder to save the output in')
    parser.add_argument("--output_template", default="dec_{region}-{date}_{jobid}")
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--use_inds", default=None, nargs="+", type=int)
    parser.add_argument("--correct_only", default=False, action="store_true")
    parser.add_argument("--include_instructed", default=False, action="store_true")
    parser.add_argument("--regions", default=None, nargs="+")
    parser.add_argument("--decoder", default="linear")
    return parser


decoder_dict = {
    "linear": {},
    "RBF": {"model": skm.SVC},
    "neighbors": {"use_nearest_neighbors": True},
}


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    data = gio.Dataset.from_readfunc(
        npa.load_gulli_hashim_data_folder,
        npa.BASEFOLDER,
        load_only_nth_files=args.use_inds,
    )
    data_use = npa.mask_completed_trials(data, correct_only=args.correct_only)
    if not args.include_instructed:
        data_use = npa.mask_uninstructed_trials(data_use)

    decoder_kwargs = decoder_dict.get(args.decoder, {})
    out_all = npra.decode_times(data_use, regions=args.regions, **decoder_kwargs)
    f, _ = npv.visualize_decoding_dict(out_all)
    if args.regions is None:
        args.regions = ("all",)

    dates = data_use["date"].to_numpy()
    
    out_fn = args.output_template.format(
        region="-".join(args.regions),
        date="-".join(dates),
        jobid=args.jobid,
    )
    
    out_fig_path = os.path.join(args.output_folder, out_fn + ".pdf")
    print(out_fig_path)
    f.savefig(out_fig_path, transparent=True, bbox_inches="tight")
    
    out_arg_path = os.path.join(args.output_folder, out_fn + ".pkl")
    pickle.dump(vars(args), open(out_arg_path, "wb"))
