# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import math
from nuscenes_uto import NuScenes
from eval.common.config import config_factory
from eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from eval.tracking.algo import TrackingEvaluation
from eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
     TrackingMetricData
from eval.tracking.loaders import create_tracks
from eval.tracking.render import recall_metric_curve, summary_plot
from eval.tracking.utils import print_final_metrics

class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 ground_file,
                 pred_file,
                 min_Dist,
                 max_Dist,
                 config: TrackingConfig,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None,
                 ):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.ground_file = ground_file
        self.pred_file = pred_file
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes
        self.min_dist_thr = min_Dist
        self.max_dist_thr = max_Dist
        # Check result file exists.
        assert os.path.exists(self.ground_file), 'Error: The ground_file  does not exist!'
        assert os.path.exists(self.pred_file), 'Error: The pred_file  does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
        self.nusc = nusc

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, self.view_name, pred_boxes, self.min_dist_thr, self.max_dist_thr, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, self.view_name, gt_boxes, self.min_dist_thr, self.max_dist_thr, self.cfg.class_range, verbose=verbose)
        self.sample_tokens = gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)
        
    def read_idsw_file(self,idsw_dirty_file):
        text = []
        with open(idsw_dirty_file,'r') as fp:
            for line in fp:
                line = line.split("\n")
                text.append(line[0].split(" "*20))

        return np.array(text)


    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        classes_infos = {}
        highglight_infos = {}
        def accumulate_class(curr_class_name):
            curr_ev = TrackingEvaluation(self.min_dist_thr,self.max_dist_thr,self.nusc,\
                                         self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,self.cfg.dist_th_tp, \
                                         self.cfg.min_recall,num_thresholds=TrackingMetricData.nelem,metric_worst=self.cfg.metric_worst, \
                                         verbose=self.verbose,output_dir=self.output_dir,render_classes=self.render_classes,
                                         )
            
            curr_md = curr_ev.accumulate()   

            classes_infos[curr_class_name] = curr_ev.GUANLIAN
            highglight_infos[curr_class_name] = curr_ev.HIGHLIGHT

            metric_data_list.set(curr_class_name, curr_md)

        for class_name in self.cfg.class_names:
            """
            bicycle/bus/car/motorcycle/pedestrian/trailer/truck
            """
            accumulate_class(class_name)

        # -----------------------------------
        # Step 2: Aggregate metrics from the metric data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        target_dict = {}
        best_score_dict = {}
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional metrics.
            # If multiple thresholds have the same value, pick the one with the highest recall.
            md = metric_data_list[class_name]  
            if np.all(np.isnan(md.motp)):   # len(md.motp) = 40
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.motp)

            if best_thresh_idx is not None:
                best_score_dict[class_name] = md.confidence[best_thresh_idx]

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            # 这里的AMOTP仅仅是指40个阈值分数对应的motp的均值，如果某阈值分数下的motp为nan的时候,就用配置配置文件tracking_nip_2019.json中的metric_wrost->amotp=2.0替代
            # motp计算了所有帧(某数据分支,val/train/test)的motp
            # 这里motp就计算了所有scene中的预测结果，没有分scene求对应的motp，然后再求所有scene的amotp
            # amota与amotp求解一致
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

            if best_thresh_idx is not None and self.save_idsw_info:   
                if not os.path.getsize(self.idsw_dirty_info_file):
                    continue
                source_texts = self.read_idsw_file(self.idsw_dirty_info_file)
                target_texts = source_texts[(source_texts[:,3] == str(md.confidence[best_thresh_idx])) & (source_texts[:,4] == class_name)]
                
                for target_text in target_texts:
                    if target_text[0] not in target_dict:
                        target_dict.setdefault(target_text[0],{})
                    if target_text[1] not in target_dict[target_text[0]]:
                        target_dict[target_text[0]].setdefault(target_text[1],[])
                    target_dict[target_text[0]][target_text[1]].append(target_text[2])

        if self.save_idsw_info:
            json_str = json.dumps(target_dict, indent=4)
            with open(self.idsw_clear_info_file,'w') as fp:
                fp.write(json_str) 
        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

                
        return metrics, metric_data_list

    def render(self, md_list: TrackingMetricDataList) -> None:
        """
        Renders a plot for each class and each metric.
        :param md_list: TrackingMetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # Plot a summary.
        summary_plot(self.cfg, md_list, savepath=savepath('summary'))

        # For each metric, plot all the classes in one diagram.
        for metric_name in LEGACY_METRICS:
            recall_metric_curve(self.cfg, md_list, metric_name, savepath=savepath('%s' % metric_name))

    def main(self, render_curves: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The serialized TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:  # true
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        # metrics_summary['meta'] = self.meta.copy()
        
        eval_result_dir = os.path.join(self.output_dir,"eval_dirs")
        if not os.path.exists(eval_result_dir):
            os.makedirs(eval_result_dir)
        with open(os.path.join(eval_result_dir, self.view_name + "_" + str(self.min_dist_thr) + "_" + str(self.max_dist_thr) + '_metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(eval_result_dir, self.view_name + "_" +  str(self.min_dist_thr) + "_" + str(self.max_dist_thr) + '_metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print metrics to stdout.
        if self.verbose:
            print_final_metrics(metrics)

        # Render curves.
        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_root', type=str,default="./tracking_results")
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/media/hjfeng/_data/code_projects/StreamPETR/StreamPETR-main/data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=0,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default='', nargs='+',
                        help='For which classes we render tracking results to disk.')
    parser.add_argument("--min_dist",type=float,default=0.0)
    parser.add_argument("--max_dist",type=float,default=-1.0)
    parser.add_argument("--gt_file",type=str,default="./data/ground_truth.json")
    parser.add_argument("--pred_file",type=str,default="./data/pred.json")   

    args = parser.parse_args()

    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')

    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialtrackize(json.load(_f))


    nusc_eval = TrackingEval(args.gt_file, args.pred_file, args.min_dist, args.max_dist, config=cfg_,eval_set=eval_set_, output_dir=output_dir_, 
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_, render_classes=render_classes_)
    
    nusc_eval.main(render_curves=render_curves_)
